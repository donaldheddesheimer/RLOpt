"""FastSAC — Spectral-RL-style SAC with staged update scheduling.

This module extends the base SAC implementation with the scheduling pattern
used in spectral-rl (spectralrl/algo/state/sac/agent.py):

  - ``feature_update_ratio``: explicit critic updates per collector iteration
  - ``actor_update_freq``: delayed actor/alpha updates (every N critic steps)
  - ``target_update_freq``: delayed target network sync (every N critic steps)

These knobs allow higher critic-to-actor update ratios, which is the core
"fast" scheduling idea.

Reference implementation:
    spectral-rl/spectralrl/algo/state/sac/agent.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import ActorCriticOperator
from torchrl.record.loggers import Logger

from rlopt.agent.sac.sac import SAC, SACConfig, SACRLOptConfig
from rlopt.config_base import NetworkConfig
from rlopt.type_aliases import OptimizerClass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class FastSACConfig(SACConfig): # acts as a wrapper for SACConfig with added scheduling knobs
    """SACConfig extended with Spectral-RL scheduling knobs.
    from spectral-rl/spectralrl/algo/state/sac/agent.py
    """

    feature_update_ratio: int | None = 1
    """Explicit upds per collector iteration. When set, overrides
    ``frames_per_batch * utd_ratio`` for computing the number of gradient
    steps.  ``None`` falls back to legacy UTD mode."""

    actor_update_freq: int = 1
    """Actor/alpha update cadence: update every N critic updates."""

    target_update_freq: int = 1
    """Target network sync cadence: sync every N critic updates."""


@dataclass
class FastSACRLOptConfig(SACRLOptConfig):
    """Top-level config for FastSAC runs."""

    sac: FastSACConfig = field(default_factory=FastSACConfig)  # type: ignore[assignment]
    """FastSAC-specific configuration."""

    def __post_init__(self):
        super().__post_init__()
        # Default experiment name distinguishes FastSAC runs on wandb dashboard
        if self.logger.exp_name == "RLOpt":
            self.logger.exp_name = "FastSAC"

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class FastSAC(SAC):
    """Soft Actor-Critic with Spectral-RL-style staged update scheduling.

    Differences from :class:`SAC`:

    * Critic is updated **every** optimisation step.
    * Actor and entropy temperature (alpha) are updated every
      ``actor_update_freq`` critic steps.
    * Target network is synced every ``target_update_freq`` critic steps.
    * ``feature_update_ratio`` directly controls the number of gradient steps
      per collector iteration, replacing the legacy ``frames_per_batch * utd``
      calculation when set.

    These three scheduling axes match the spectral-rl SAC agent
    (``spectralrl/algo/state/sac/agent.py``).
    """

    def __init__(
        self,
        env,
        config: FastSACRLOptConfig,
        logger: Logger | None = None,
        **kwargs,
    ):
        self._normalize_network_input_keys(config)
        super().__init__(
            env=env,
            config=config,
            logger=logger,
            **kwargs,
        )

        self.config = cast(FastSACRLOptConfig, self.config)
        self.config: FastSACRLOptConfig

        # Counters for staged scheduling verification
        self.total_actor_updates = 0
        self.total_target_updates = 0

    @staticmethod
    def _normalize_nested_keys(
        keys: list[Any] | None,
    ) -> list[str | tuple[str, ...]] | None:
        if keys is None:
            return None

        normalized: list[str | tuple[str, ...]] = []
        for key in keys:
            if isinstance(key, str):
                normalized.append(key)
                continue

            try:
                nested_key = tuple(key)
            except TypeError as exc:
                raise ValueError(
                    "FastSAC input_keys entries must be strings or sequences of strings."
                ) from exc

            if len(nested_key) == 0 or not all(
                isinstance(part, str) for part in nested_key
            ):
                raise ValueError(
                    "FastSAC input_keys nested entries must be non-empty sequences of strings."
                )

            normalized.append(nested_key)

        return normalized

    @classmethod
    def _normalize_network_input_keys(cls, config: FastSACRLOptConfig) -> None:
        for network_name in ("policy", "q_function", "value_function"):
            network_cfg = getattr(config, network_name, None)
            if not isinstance(network_cfg, NetworkConfig):
                continue
            normalized = cls._normalize_nested_keys(network_cfg.input_keys)
            if normalized is not None:
                network_cfg.input_keys = cast(list[str], normalized)

    def _construct_actor_critic(self) -> TensorDictModule:
        if self.q_function is None or self.policy is None:
            msg = "SAC requires a Q-function and policy configuration."
            raise ValueError(msg)

        if self.feature_extractor:
            return ActorCriticOperator(
                common_operator=self.feature_extractor,
                policy_operator=self.policy,
                value_operator=self.q_function,
            )

        class NoOpModule(torch.nn.Module):
            def forward(self):
                return ()

        dummy = TensorDictModule(
            module=NoOpModule(),
            in_keys=[],
            out_keys=[],
        )
        return ActorCriticOperator(
            common_operator=dummy,
            policy_operator=self.policy,
            value_operator=self.q_function,
        )

    # ------------------------------------------------------------------
    # Optimizers — store individual references for staged stepping
    # ------------------------------------------------------------------

    def _set_optimizers(
        self, optimizer_cls: OptimizerClass, optimizer_kwargs: dict[str, Any]
    ) -> list[torch.optim.Optimizer]:
        optimizers = super()._set_optimizers(optimizer_cls, optimizer_kwargs)
        # optimizers order: [actor, critic, (alpha)]
        self._actor_optim = optimizers[0]
        self._critic_optim = optimizers[1]
        self._alpha_optim = optimizers[2] if len(optimizers) > 2 else None
        return optimizers

    # ------------------------------------------------------------------
    # Staged update  (mirrors spectral-rl train_step)
    # ------------------------------------------------------------------

    def update(self, sampled_tensordict: TensorDict) -> TensorDict:
        """Single gradient step with staged actor/target scheduling.

        Critic loss is always back-propagated and stepped.  Actor + alpha
        losses are only back-propagated when ``total_network_updates`` is
        a multiple of ``actor_update_freq``.  Target sync happens when
        ``total_network_updates`` is a multiple of ``target_update_freq``.

        Reference: spectral-rl/spectralrl/algo/state/sac/agent.py  train_step
        """
        sac_cfg = self.config.sac
        self.total_network_updates += 1

        do_actor = (self.total_network_updates % sac_cfg.actor_update_freq == 0)
        do_target = (self.total_network_updates % sac_cfg.target_update_freq == 0)

        kl_context = None
        policy_op = None
        if (self.config.optim.scheduler or "").lower() == "adaptive":
            policy_op = self.actor_critic.get_policy_operator()
            kl_context = self._prepare_kl_context(sampled_tensordict, policy_op)

        sampled_tensordict = self._ensure_old_policy_info(sampled_tensordict)

        # Compute all losses via the loss module
        loss_td = self.loss_module(sampled_tensordict)

        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        alpha_loss = loss_td["loss_alpha"]

        # --- Staged backward: critic always, actor + alpha when scheduled ---
        if do_actor:
            (actor_loss + q_loss + alpha_loss).sum().backward()  # type: ignore[operator]
        else:
            q_loss.sum().backward()

        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_norm=1.0)

        # --- Critic update (always) ---
        self._critic_optim.step()
        self._critic_optim.zero_grad(set_to_none=True)

        # --- Actor + alpha update (gated) ---
        if do_actor:
            self._actor_optim.step()
            self._actor_optim.zero_grad(set_to_none=True)
            if self._alpha_optim is not None:
                self._alpha_optim.step()
                self._alpha_optim.zero_grad(set_to_none=True)
            self.total_actor_updates += 1

            if kl_context is not None and policy_op is not None:
                kl_approx = self._compute_kl_after_update(kl_context, policy_op)
                if kl_approx is not None:
                    loss_td.set("kl_approx", kl_approx.detach())
                    self._maybe_adjust_lr(kl_approx, self.config.optim)

        # --- Target update (gated) ---
        if do_target:
            self.target_net_updater.step()
            self.total_target_updates += 1

        return loss_td.detach_()

    # ------------------------------------------------------------------
    # Train loop — respect feature_update_ratio
    # ------------------------------------------------------------------

    def train(self) -> None:  # type: ignore[override]
        """Train loop that uses ``feature_update_ratio`` when set.

        When ``feature_update_ratio`` is not ``None`` (the FastSAC default),
        it directly controls the number of gradient steps per collector
        iteration.  Otherwise falls back to the legacy
        ``frames_per_batch * utd_ratio`` calculation.

        Reference: spectral-rl/spectralrl/algo/state/sac/agent.py  train_step
        """
        # Temporarily patch the UTD-derived num_updates if feature_update_ratio
        # is set.  We achieve this by adjusting utd_ratio so the parent's
        # train() computes the right number of updates.
        cfg = self.config
        sac_cfg = cfg.sac
        prev_utd: float | None = None
        if sac_cfg.feature_update_ratio is not None:
            # Parent computes: num_updates = int(frames_per_batch * utd_ratio)
            # We want: num_updates = feature_update_ratio
            prev_utd = float(sac_cfg.utd_ratio)
            fpb = cfg.collector.frames_per_batch
            if fpb > 0:
                sac_cfg.utd_ratio = float(sac_cfg.feature_update_ratio) / float(fpb)
            else:
                sac_cfg.utd_ratio = 1.0

        try:
            super().train()
        finally:
            if prev_utd is not None:
                sac_cfg.utd_ratio = prev_utd
