"""IPMDFastSAC: FastSAC (C51 off-policy) with IPMD inverse-reward estimator.

The policy is optimized entirely by FastSAC's distributional off-policy update
(C51 critic, replay buffer, aggressive tau). The IPMD reward model is trained
alongside and its estimated rewards are blended into the replay-buffer rewards
at collect time.

Architecture:
  - Actor/Critic: FastSAC (unchanged)
  - Replay buffer: SimpleReplayBuffer storing augmented rewards
  - Reward model: MLP trained with IPMD objective (r_pi - r_exp + L2 + grad_penalty)
  - Expert sampler: attached from env.sample_expert_batch() via BaseAlgorithm helper
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torchrl._utils import timeit
from torchrl.modules import MLP
from torchrl.record.loggers import Logger

from rlopt.agent.ipmd.ipmd import IPMDConfig
from rlopt.agent.sac.fastsac import (
    EmpiricalNormalization,
    FastSAC,
    FastSACIterationData,
    FastSACRLOptConfig,
    FastSACTrainingMetadata,
)
from rlopt.config_utils import dedupe_keys, flatten_feature_tensor, next_obs_key
from rlopt.utils import get_activation_class

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class IPMDFastSACRLOptConfig(FastSACRLOptConfig):
    """FastSAC config extended with IPMD reward-model settings.

    All FastSAC fields (fastsac.*) govern the actor/critic/replay loop.
    All IPMD fields (ipmd.*) govern the reward estimator.

    Note: IPMDConfig inherits from PPOConfig for historical reasons, but only
    the reward-specific fields are used here — PPO-specific fields are ignored.
    """

    ipmd: IPMDConfig = field(default_factory=IPMDConfig)
    """IPMD reward-model configuration (only reward_* fields are used)."""


# ---------------------------------------------------------------------------
# IPMDFastSAC algorithm
# ---------------------------------------------------------------------------


class IPMDFastSAC(FastSAC):
    """FastSAC with IPMD inverse-reward augmentation.

    Policy optimization uses the full FastSAC pipeline (off-policy, C51
    distributional critic, SimpleReplayBuffer). An IPMD reward model
    r(s, a, s') is trained separately and its output is blended into the
    replay-buffer rewards at collect time::

        reward_stored = r_env + est_reward_weight * r_est.clamp(min, max)

    The reward model is trained with the IPMD objective each iteration::

        loss = reward_loss_coeff * (r_pi.mean() - r_exp.mean())
             + reward_l2_coeff  * sqrt(r_pi^2.mean() + r_exp^2.mean())
             + reward_grad_penalty_coeff * grad_penalty

    Raw rollout TensorDicts (reward-obs keys only) are cached in a small
    deque so the reward model always has recent policy data to train on.
    """

    def __init__(
        self,
        env,
        config: IPMDFastSACRLOptConfig,
        logger: Logger | None = None,
        **kwargs: Any,
    ) -> None:
        # Pre-init bookkeeping — must exist before super() runs any method that
        # might call _estimate_reward or similar.
        self._reward_raw_buffer: deque[TensorDict] = deque(maxlen=20)
        self._expert_batch_sampler = None

        # Build actor, critic, replay buffer (FastSAC.__init__)
        super().__init__(env=env, config=config, logger=logger, **kwargs)

        # Build reward model *after* super() so self.env and self.device are set
        self._init_reward_model()

        # Discover and wrap env.sample_expert_batch — inherited helper from BaseAlgorithm
        self._auto_attach_env_expert_sampler()

    # ------------------------------------------------------------------
    # Reward model initialisation
    # ------------------------------------------------------------------

    def _init_reward_model(self) -> None:
        """Build the reward-estimator MLP and its optimizer."""
        cfg = self.config
        assert isinstance(cfg, IPMDFastSACRLOptConfig), (
            "IPMDFastSAC requires IPMDFastSACRLOptConfig, "
            f"got {type(cfg).__name__!r}"
        )
        ipmd = cfg.ipmd

        # --- Reward obs keys --------------------------------------------------
        reward_keys = ipmd.reward_input_keys
        if not reward_keys:
            reward_keys = cfg.policy.get_input_keys()
        self._reward_obs_keys: list = dedupe_keys(list(reward_keys))

        # Feature dim cache (needed by _obs_features_from_td)
        self._obs_feature_dims: dict = {}
        self._obs_feature_ndims: dict = {}
        for key in self._reward_obs_keys:
            shape = self._obs_key_feature_shape(key)
            self._obs_feature_ndims[key] = len(shape)
            self._obs_feature_dims[key] = int(math.prod(shape)) if shape else 1

        # Action feature dim
        action_spec = getattr(self.env, "action_spec_unbatched", self.env.action_spec)
        action_shape = tuple(int(d) for d in action_spec.shape)
        self._action_feature_ndim: int = len(action_shape)
        self._action_feature_dim: int = int(math.prod(action_shape)) if action_shape else 1

        # --- Reward input-type flags -------------------------------------------
        rit = ipmd.reward_input_type
        _valid_rit = {"s", "s'", "sa", "sas"}
        if rit not in _valid_rit:
            raise ValueError(
                f"reward_input_type must be one of {sorted(_valid_rit)}, got {rit!r}"
            )
        self._rit_use_s: bool = rit in ("s", "sa", "sas")
        self._rit_use_a: bool = rit in ("sa", "sas")
        self._rit_use_sn: bool = rit in ("s'", "sas")

        # --- Scalar hyper-parameter caches ------------------------------------
        self._reward_loss_coeff: float = float(ipmd.reward_loss_coeff)
        self._reward_l2_coeff: float = float(ipmd.reward_l2_coeff)
        self._reward_grad_penalty_coeff: float = float(ipmd.reward_grad_penalty_coeff)
        self._reward_detach_features: bool = bool(ipmd.reward_detach_features)
        self._est_reward_weight: float = float(ipmd.est_reward_weight)
        self._est_reward_clamp_min = ipmd.estimated_reward_clamp_min
        self._est_reward_clamp_max = ipmd.estimated_reward_clamp_max

        # --- Output activation ------------------------------------------------
        out_act = ipmd.reward_output_activation
        scale = float(ipmd.reward_output_scale)
        if out_act == "tanh":
            self._reward_out_fn = lambda r: torch.tanh(r) * scale
        elif out_act == "sigmoid":
            self._reward_out_fn = lambda r: torch.sigmoid(r) * scale
        else:
            self._reward_out_fn = lambda r: r

        # --- Build reward estimator -------------------------------------------
        obs_dim = sum(self._obs_feature_dims[k] for k in self._reward_obs_keys)
        act_dim = self._action_feature_dim

        if rit in ("s", "s'"):
            in_dim = obs_dim
        elif rit == "sa":
            in_dim = obs_dim + act_dim
        else:  # "sas"
            in_dim = obs_dim * 2 + act_dim

        self.reward_estimator = MLP(
            in_features=in_dim,
            out_features=1,
            num_cells=list(ipmd.reward_num_cells),
            activation_class=get_activation_class(ipmd.reward_activation),
            device=self.device,
        )

        self._reward_optim = torch.optim.Adam(
            self.reward_estimator.parameters(),
            lr=cfg.optim.lr,
            weight_decay=0.0,
        )

    # ------------------------------------------------------------------
    # Observation utilities (mirrors IPMD without PPO dependency)
    # ------------------------------------------------------------------

    def _obs_key_feature_shape(self, key) -> tuple[int, ...]:
        """Return unbatched feature shape for *key* from the env observation spec."""
        shape = tuple(int(d) for d in self.env.observation_spec[key].shape)
        batch_prefix = tuple(int(d) for d in getattr(self.env, "batch_size", ()))
        if (
            len(batch_prefix) > 0
            and len(shape) >= len(batch_prefix)
            and shape[: len(batch_prefix)] == batch_prefix
        ):
            return shape[len(batch_prefix):]
        return shape

    def _obs_features_from_td(
        self,
        td: TensorDict,
        keys: list,
        *,
        next_obs: bool,
        detach: bool,
    ) -> Tensor:
        """Concatenate observation features from *td*, optionally from 'next' slot."""
        parts: list[Tensor] = []
        for key in keys:
            obs = flatten_feature_tensor(
                td.get(next_obs_key(key) if next_obs else key),
                self._obs_feature_ndims[key],
            )
            parts.append(obs.detach() if detach else obs)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _reward_input_from_td(
        self,
        td: TensorDict,
        *,
        detach: bool = True,
        requires_grad: bool = False,
    ) -> Tensor:
        """Assemble the reward-estimator input tensor from a TensorDict.

        Builds (s, a, s') according to ``reward_input_type`` flags.
        """
        parts: list[Tensor] = []
        if self._rit_use_s:
            parts.append(
                self._obs_features_from_td(
                    td, self._reward_obs_keys, next_obs=False, detach=detach
                )
            )
        if self._rit_use_a:
            action = flatten_feature_tensor(td.get("action"), self._action_feature_ndim)
            parts.append(action.detach() if detach else action)
        if self._rit_use_sn:
            parts.append(
                self._obs_features_from_td(
                    td, self._reward_obs_keys, next_obs=True, detach=detach
                )
            )
        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        if requires_grad:
            x = x.detach().requires_grad_(True)
        return x

    @torch.no_grad()
    def _estimate_reward(self, td: TensorDict) -> Tensor:
        """Compute estimated reward [batch] from a TensorDict (no grad)."""
        x = self._reward_input_from_td(td, detach=True)
        return self._reward_out_fn(self.reward_estimator(x)).squeeze(-1)

    def _reward_model_update_enabled(self) -> bool:
        """Return True if the reward model loss has any non-zero coefficient."""
        return (
            self._reward_loss_coeff != 0.0
            or self._reward_l2_coeff > 0.0
            or self._reward_grad_penalty_coeff > 0.0
        )

    # ------------------------------------------------------------------
    # Expert batch utilities
    # ------------------------------------------------------------------

    def _expert_required_keys(self) -> list:
        """Keys the expert sampler must provide for reward model training."""
        required: list = []
        if self._rit_use_s:
            required.extend(self._reward_obs_keys)
        if self._rit_use_sn:
            required.extend(next_obs_key(k) for k in self._reward_obs_keys)
        if self._rit_use_a:
            required.append("action")
        return dedupe_keys(required)

    def _next_expert_batch(self, batch_size: int | None = None) -> TensorDict:
        """Sample a batch from the expert sampler attached to this agent."""
        cfg = self.config
        assert isinstance(cfg, IPMDFastSACRLOptConfig)
        effective_bs = int(
            batch_size
            or cfg.ipmd.expert_batch_size
            or cfg.fastsac.batch_size * cfg.env.num_envs
        )
        if self._expert_batch_sampler is None:
            raise RuntimeError(
                "IPMDFastSAC requires env.sample_expert_batch(). "
                "The expert sampler was not attached — check env setup."
            )
        required_keys = self._expert_required_keys()
        expert_batch = self._expert_batch_sampler(effective_bs, required_keys)
        if expert_batch is None:
            raise RuntimeError("Expert sampler returned None.")
        if expert_batch.numel() > effective_bs:
            expert_batch = expert_batch[:effective_bs]
        return expert_batch.to(self.device)

    # ------------------------------------------------------------------
    # collect() — augment rewards with r_est before storing in replay buf
    # ------------------------------------------------------------------

    def collect(
        self, metadata: FastSACTrainingMetadata, iteration_idx: int
    ) -> FastSACIterationData:
        """Collect one step, blend IPMD reward into env reward, cache for reward model.

        Overrides FastSAC.collect() to:
        1. Compute r_est from the reward estimator (no_grad).
        2. Store ``r_env + est_reward_weight * r_est.clamp(...)`` in the replay buffer.
        3. Cache a small reward-obs TensorDict for later reward model training.
        """
        cfg = self.config
        fsac = cfg.fastsac
        num_envs = cfg.env.num_envs

        with timeit("collect"):
            data = next(metadata.collector_iter)

        self.collector.update_policy_weights_()

        frames = data.numel()
        metadata.frames_processed += frames
        metadata.progress_bar.update(frames)

        steps_per_env = max(1, frames // num_envs)
        flat_data = data.reshape(-1) if steps_per_env > 1 else data

        # Standard FastSAC obs extraction
        actor_obs = self._extract_obs(flat_data, self._actor_input_keys)
        next_actor_obs = self._extract_next_obs(flat_data, self._actor_input_keys)
        critic_obs = self._extract_obs(flat_data, self._critic_input_keys)
        next_critic_obs = self._extract_next_obs(flat_data, self._critic_input_keys)

        if fsac.norm_obs and isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer(actor_obs, update=True)
            self.critic_obs_normalizer(critic_obs, update=True)

        # Handle terminal obs replacement (truncation)
        truncations = flat_data["next", "truncated"].reshape(frames)
        if ("next", "obs_unbatched") in flat_data.keys(True):
            term_td = flat_data["next", "obs_unbatched"]
            term_actor_obs = self._extract_obs(term_td, self._actor_input_keys)
            term_critic_obs = self._extract_obs(term_td, self._critic_input_keys)
            mask = truncations.bool().unsqueeze(-1)
            next_actor_obs = torch.where(mask, term_actor_obs, next_actor_obs)
            next_critic_obs = torch.where(mask, term_critic_obs, next_critic_obs)

        actions = flat_data["action"].reshape(frames, -1)
        env_rewards = flat_data["next", "reward"].reshape(frames)
        dones = flat_data["next", "done"].reshape(frames).long()
        trunc_long = truncations.long()

        # --- IPMD reward augmentation -----------------------------------------
        rewards = env_rewards
        if self._reward_model_update_enabled():
            try:
                with torch.no_grad():
                    self.reward_estimator.eval()
                    r_est = self._estimate_reward(flat_data)
                    self.reward_estimator.train()
                if r_est.ndim != 1:
                    r_est = r_est.reshape(-1)
                if r_est.shape != env_rewards.shape:
                    logger.warning(
                        "Skipping reward augmentation: estimated reward shape %s does not match env reward shape %s.",
                        tuple(r_est.shape),
                        tuple(env_rewards.shape),
                    )
                    r_est = None
                if r_est is not None:
                    r_est = r_est.clamp(
                        min=self._est_reward_clamp_min,
                        max=self._est_reward_clamp_max,
                    )
                    rewards = env_rewards + self._est_reward_weight * r_est
            except (KeyError, RuntimeError):
                # Reward obs keys not yet available (warm-up) — fall back to env reward
                pass

        # --- Cache raw obs for reward model training --------------------------
        if self._reward_model_update_enabled():
            select_keys: list = []
            if self._rit_use_s:
                select_keys.extend(self._reward_obs_keys)
            if self._rit_use_sn:
                for k in self._reward_obs_keys:
                    select_keys.append(("next", *k) if isinstance(k, tuple) else ("next", k))
            if self._rit_use_a:
                select_keys.append("action")
            try:
                raw_cache = flat_data.select(*select_keys).detach().cpu()
                self._reward_raw_buffer.append(raw_cache)
            except KeyError:
                pass

        # --- Extend replay buffer (augmented rewards) -------------------------
        with timeit("replay_extend"):
            for t in range(steps_per_env):
                s, e = t * num_envs, (t + 1) * num_envs
                transition = TensorDict(
                    {
                        "observations": actor_obs[s:e],
                        "actions": actions[s:e],
                        "next": {
                            "observations": next_actor_obs[s:e],
                            "rewards": rewards[s:e],
                            "dones": dones[s:e],
                            "truncations": trunc_long[s:e],
                        },
                    },
                    batch_size=[num_envs],
                    device=self.device,
                )
                if self._obs_dim != self._critic_obs_dim:
                    transition["critic_observations"] = critic_obs[s:e]
                    transition["next"]["critic_observations"] = next_critic_obs[s:e]
                self.data_buffer.extend(transition)

        return FastSACIterationData(
            iteration_idx=iteration_idx,
            frames=frames,
            rollout=data,
        )

    # ------------------------------------------------------------------
    # Reward model update (IPMD objective)
    # ------------------------------------------------------------------

    def _update_reward_estimator(self) -> dict[str, float]:
        """Train the reward estimator for one step using IPMD objective.

        Loss = reward_loss_coeff * (r_pi.mean() - r_exp.mean())
             + reward_l2_coeff  * sqrt(r_pi^2.mean() + r_exp^2.mean())
             + reward_grad_penalty_coeff * gradient_penalty
        """
        if not self._reward_model_update_enabled():
            return {}
        if len(self._reward_raw_buffer) == 0:
            return {}
        if self._expert_batch_sampler is None:
            return {}

        # Sample a random cached rollout entry for policy side
        buf_list = list(self._reward_raw_buffer)
        idx = torch.randint(len(buf_list), (1,)).item()
        policy_td = buf_list[int(idx)].to(self.device)

        # Sample expert batch
        try:
            expert_td = self._next_expert_batch()
        except RuntimeError:
            return {}

        needs_grad_penalty = self._reward_grad_penalty_coeff > 0.0

        # Build reward inputs (grad required for penalty)
        r_pi_input = self._reward_input_from_td(
            policy_td, detach=True, requires_grad=needs_grad_penalty
        )
        r_exp_input = self._reward_input_from_td(
            expert_td, detach=True, requires_grad=needs_grad_penalty
        )

        self.reward_estimator.train()
        r_pi = self._reward_out_fn(self.reward_estimator(r_pi_input)).squeeze(-1)
        r_exp = self._reward_out_fn(self.reward_estimator(r_exp_input)).squeeze(-1)

        diff = r_pi.mean() - r_exp.mean()
        l2 = r_pi.pow(2).mean() + r_exp.pow(2).mean()

        grad_penalty = torch.zeros((), device=self.device)
        if needs_grad_penalty:
            for r_val, x_input in [(r_pi, r_pi_input), (r_exp, r_exp_input)]:
                grads = torch.autograd.grad(
                    outputs=r_val.sum(),
                    inputs=x_input,
                    create_graph=True,
                    only_inputs=True,
                )[0]
                grad_penalty = grad_penalty + grads.pow(2).sum(dim=-1).mean()

        loss = (
            self._reward_loss_coeff * diff
            + self._reward_l2_coeff * l2.pow(0.5)
            + self._reward_grad_penalty_coeff * grad_penalty
        )

        self._reward_optim.zero_grad(set_to_none=True)
        loss.backward()
        max_grad = float(getattr(self.config.optim, "max_grad_norm", 0) or 0)
        if max_grad > 0:
            clip_grad_norm_(self.reward_estimator.parameters(), max_grad)
        self._reward_optim.step()

        return {
            "ipmd/reward_loss": loss.detach().item(),
            "ipmd/reward_diff": diff.detach().item(),
            "ipmd/reward_l2": float(l2.detach().sqrt().item()),
            "ipmd/r_pi_mean": r_pi.detach().mean().item(),
            "ipmd/r_exp_mean": r_exp.detach().mean().item(),
        }

    # ------------------------------------------------------------------
    # iterate() — FastSAC update + IPMD reward model update
    # ------------------------------------------------------------------

    def iterate(
        self, iteration: FastSACIterationData, metadata: FastSACTrainingMetadata
    ) -> None:
        """Run FastSAC gradient updates then update the IPMD reward model."""
        super().iterate(iteration, metadata)
        reward_metrics = self._update_reward_estimator()
        iteration.metrics.update(reward_metrics)

    # ------------------------------------------------------------------
    # Progress display
    # ------------------------------------------------------------------

    def _progress_summary_fields(self) -> tuple[tuple[str, str], ...]:
        return (
            *super()._progress_summary_fields(),
            ("ipmd/reward_diff", "rwd_diff"),
            ("ipmd/r_pi_mean", "r_pi"),
            ("ipmd/r_exp_mean", "r_exp"),
        )
