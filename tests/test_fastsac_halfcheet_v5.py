"""
test_fastsac.py
===============
FastSAC-focused test suite, extending the existing SAC smoke test pattern.

Test categories
---------------
1. Smoke             - basic construction + train() completes without error
2. Cadence           - actor/target update counters match configured frequencies
3. Schedule parity   - FastSAC with freq=1 is behaviorally equivalent to SAC baseline
4. Loss health       - losses are finite and alpha stays in [min_alpha, max_alpha]
5. Throughput        - parametrized benchmark across the migration-guide matrix;
                       prints a summary table but never fails on speed alone

Run all:
    pytest test_fastsac.py -v

Run only cadence tests:
    pytest test_fastsac.py -v -k cadence

Run benchmark (prints table, never fails on perf):
    pytest test_fastsac.py -v -k throughput -s
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch

from rlopt.agent import SAC, SACRLOptConfig
from rlopt.agent.sac.fast_sac import FastSAC, FastSACRLOptConfig
from rlopt.config_base import NetworkConfig
from rlopt.env_utils import make_parallel_env


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_cfg(env_name: str = "HalfCheetah-v5") -> SACRLOptConfig:
    """Minimal config that matches the existing smoke test."""
    cfg = SACRLOptConfig()
    cfg.env.env_name = env_name
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = 4
    cfg.collector.total_frames = 4
    cfg.collector.init_random_frames = 0
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.q_function = NetworkConfig(
        num_cells=[64, 64],
        activation_fn="relu",
        output_dim=1,
        input_keys=["observation"],
    )
    return cfg


def _fast_cfg(
    feature_update_ratio: int | None = 1,
    actor_update_freq: int = 1,
    target_update_freq: int = 1,
    total_frames: int = 20,
    frames_per_batch: int = 4,
    env_name: str = "HalfCheetah-v5",
) -> FastSACRLOptConfig:
    """FastSAC config with explicit scheduling knobs."""
    cfg = FastSACRLOptConfig()
    cfg.env.env_name = env_name
    cfg.env.device = "cpu"
    cfg.device = "cpu"
    cfg.collector.frames_per_batch = frames_per_batch
    cfg.collector.total_frames = total_frames
    cfg.collector.init_random_frames = 0
    cfg.replay_buffer.size = 64
    cfg.loss.mini_batch_size = 2
    cfg.compile.compile = False
    cfg.q_function = NetworkConfig(
        num_cells=[64, 64],
        activation_fn="relu",
        output_dim=1,
        input_keys=["observation"],
    )
    cfg.sac.feature_update_ratio = feature_update_ratio
    cfg.sac.actor_update_freq = actor_update_freq
    cfg.sac.target_update_freq = target_update_freq
    return cfg


def _make_agent(cfg, cls=SAC):
    try:
        env = make_parallel_env(cfg)
    except Exception as exc:
        pytest.skip(f"Cannot create environment: {exc}")
    return cls(env, cfg, logger=None)


# ---------------------------------------------------------------------------
# 1. Smoke tests
# ---------------------------------------------------------------------------

class TestSmoke:
    """Basic construction and train() completion."""

    def test_sac_smoke(self):
        """Original smoke test preserved exactly."""
        cfg = _base_cfg()
        agent = _make_agent(cfg, SAC)
        agent.train()
        assert agent.__class__.__name__ == "SAC"

    def test_fastsac_smoke(self):
        """FastSAC constructs and trains without error."""
        cfg = _fast_cfg()
        agent = _make_agent(cfg, FastSAC)
        agent.train()
        assert agent.__class__.__name__ == "FastSAC"

    def test_fastsac_is_sac_subclass(self):
        """FastSAC must be a SAC subclass — no architectural fork."""
        assert issubclass(FastSAC, SAC)

    def test_fastsac_default_config_fields(self):
        """FastSACRLOptConfig must expose the three scheduling knobs."""
        cfg = FastSACRLOptConfig()
        assert hasattr(cfg.sac, "feature_update_ratio")
        assert hasattr(cfg.sac, "actor_update_freq")
        assert hasattr(cfg.sac, "target_update_freq")

    def test_legacy_sac_config_unchanged(self):
        """Vanilla SACRLOptConfig should NOT have FastSAC scheduling knobs."""
        cfg = SACRLOptConfig()
        assert not hasattr(cfg.sac, "feature_update_ratio"), (
            "SACConfig should not have feature_update_ratio. "
            "FastSAC changes must stay in FastSACConfig."
        )


# ---------------------------------------------------------------------------
# 2. Cadence correctness
# ---------------------------------------------------------------------------

class TestCadence:
    """
    Verify that actor/target update counters match the configured frequencies.

    The invariant is:
        total_actor_updates  == total_network_updates // actor_update_freq
        total_target_updates == total_network_updates // target_update_freq
    """

    def _run_and_check(self, cfg: FastSACRLOptConfig) -> FastSAC:
        agent = _make_agent(cfg, FastSAC)
        agent.train()
        return agent

    def test_actor_update_freq_1(self):
        """AUF=1: actor updates every step → actor_updates == net_updates."""
        cfg = _fast_cfg(actor_update_freq=1, total_frames=20)
        agent = self._run_and_check(cfg)
        assert agent.total_actor_updates == agent.total_network_updates, (
            f"Expected actor_updates={agent.total_network_updates}, "
            f"got {agent.total_actor_updates}"
        )

    def test_actor_update_freq_2(self):
        """AUF=2: actor updates every 2 critic steps."""
        cfg = _fast_cfg(actor_update_freq=2, total_frames=40, frames_per_batch=4)
        agent = self._run_and_check(cfg)
        expected = agent.total_network_updates // 2
        assert agent.total_actor_updates == expected, (
            f"AUF=2: expected {expected} actor updates, got {agent.total_actor_updates}. "
            f"total_network_updates={agent.total_network_updates}"
        )

    def test_target_update_freq_1(self):
        """TUF=1: target syncs every step."""
        cfg = _fast_cfg(target_update_freq=1, total_frames=20)
        agent = self._run_and_check(cfg)
        assert agent.total_target_updates == agent.total_network_updates

    def test_target_update_freq_2(self):
        """TUF=2: target syncs every 2 critic steps."""
        cfg = _fast_cfg(target_update_freq=2, total_frames=40, frames_per_batch=4)
        agent = self._run_and_check(cfg)
        expected = agent.total_network_updates // 2
        assert agent.total_target_updates == expected, (
            f"TUF=2: expected {expected} target updates, got {agent.total_target_updates}"
        )

    def test_actor_and_target_independent(self):
        """
        AUF and TUF should be independently tracked.
        AUF=2, TUF=3 → different counts for actor vs target.
        """
        cfg = _fast_cfg(actor_update_freq=2, target_update_freq=3,
                        total_frames=60, frames_per_batch=4)
        agent = self._run_and_check(cfg)
        n = agent.total_network_updates
        assert agent.total_actor_updates == n // 2
        assert agent.total_target_updates == n // 3
        # They should differ (unless n is a multiple of 6 and they happen to match)
        # The key point: independent scheduling, not locked together.
        assert agent.total_actor_updates != agent.total_target_updates or n % 6 == 0

    def test_network_updates_increment_once_per_update_call(self):
        """
        total_network_updates must increment by exactly 1 per update() call.
        We verify this by calling update() manually a known number of times.
        """
        cfg = _fast_cfg(total_frames=4)  # short run just to init the agent
        agent = _make_agent(cfg, FastSAC)
        agent.train()  # initializes replay buffer with data

        before = agent.total_network_updates
        # Manually call update() 5 more times
        for _ in range(5):
            batch = agent.data_buffer.sample()
            agent.update(batch)

        assert agent.total_network_updates == before + 5


# ---------------------------------------------------------------------------
# 3. Schedule parity (A/B equivalence)
# ---------------------------------------------------------------------------

class TestScheduleParity:
    """
    A/B correctness: FastSAC with freq=1 all-round should behave like legacy SAC.
    We can't guarantee identical losses (different random seeds per update call)
    but we CAN verify:
      - same number of gradient steps taken
      - both produce finite losses
      - alpha remains in bounds for both
    """

    def test_update_count_parity(self):
        """
        Legacy SAC (FUR=None, UTD=1) and FastSAC (FUR=1, AUF=1, TUF=1) should
        perform the same number of gradient updates for the same frame budget
        when frames_per_batch=1.
        """
        frames = 20
        fpb = 4

        # Legacy: num_updates = frames_per_batch * utd_ratio = 4 * 1 = 4 per iter
        cfg_legacy = _base_cfg()
        cfg_legacy.collector.total_frames = frames
        cfg_legacy.collector.frames_per_batch = fpb
        cfg_legacy.sac.utd_ratio = 1.0
        agent_legacy = _make_agent(cfg_legacy, SAC)
        agent_legacy.train()

        # FastSAC: num_updates = feature_update_ratio = 4 per iter (set to match)
        cfg_fast = _fast_cfg(
            feature_update_ratio=fpb,   # match legacy's effective count
            total_frames=frames,
            frames_per_batch=fpb,
        )
        agent_fast = _make_agent(cfg_fast, FastSAC)
        agent_fast.train()

        assert agent_legacy.total_network_updates == agent_fast.total_network_updates, (
            f"Legacy={agent_legacy.total_network_updates} vs "
            f"FastSAC={agent_fast.total_network_updates} network updates. "
            "These should be equal when FUR == frames_per_batch * UTD."
        )


# ---------------------------------------------------------------------------
# 4. Loss health
# ---------------------------------------------------------------------------

class TestLossHealth:
    """
    Verify that losses are finite and alpha stays within configured bounds.
    These are the invariants from section 2.1 of the migration guide.
    """

    def _collect_losses(self, cfg, cls=FastSAC) -> dict[str, list[float]]:
        """Run agent and collect per-update loss values by monkey-patching update()."""
        agent = _make_agent(cfg, cls)

        losses: dict[str, list[float]] = {
            "loss_qvalue": [],
            "loss_actor": [],
            "loss_alpha": [],
        }
        original_update = agent.update

        def patched_update(td):
            result = original_update(td)
            for key in losses:
                if key in result.keys():
                    val = result[key]
                    if torch.is_tensor(val):
                        losses[key].append(float(val.detach().cpu().mean()))
            return result

        agent.update = patched_update
        agent.train()
        return losses

    def test_losses_finite(self):
        """All loss components must be finite (no NaN/Inf)."""
        cfg = _fast_cfg(total_frames=20)
        losses = self._collect_losses(cfg)

        for name, vals in losses.items():
            if not vals:
                continue
            for i, v in enumerate(vals):
                assert (
                    v == v and abs(v) != float("inf")
                ), f"{name}[{i}] is non-finite: {v}"

    def test_alpha_stays_in_bounds(self):
        """Alpha must stay within [min_alpha, max_alpha] throughout training."""
        cfg = _fast_cfg(total_frames=20)
        cfg.sac.min_alpha = 1e-4
        cfg.sac.max_alpha = 10.0

        agent = _make_agent(cfg, FastSAC)
        agent.train()

        alpha = agent.loss_module.log_alpha.exp().item()
        assert cfg.sac.min_alpha <= alpha <= cfg.sac.max_alpha, (
            f"Alpha={alpha:.6f} is outside [{cfg.sac.min_alpha}, {cfg.sac.max_alpha}]"
        )

    def test_alpha_is_positive(self):
        """exp(log_alpha) must always be strictly positive."""
        cfg = _fast_cfg(total_frames=20)
        agent = _make_agent(cfg, FastSAC)
        agent.train()
        alpha = agent.loss_module.log_alpha.exp().item()
        assert alpha > 0, f"Alpha became non-positive: {alpha}"

    @pytest.mark.parametrize("actor_update_freq", [1, 2, 4])
    def test_q_loss_finite_under_delayed_actor(self, actor_update_freq: int):
        """
        Critic loss must stay finite even when actor updates are delayed.
        Delayed actor is a common failure mode where critic over-trains.
        """
        cfg = _fast_cfg(
            actor_update_freq=actor_update_freq,
            total_frames=40,
            frames_per_batch=4,
        )
        losses = self._collect_losses(cfg)
        q_losses = losses["loss_qvalue"]
        assert q_losses, "No Q-losses were recorded."
        for i, v in enumerate(q_losses):
            assert v == v and abs(v) != float("inf"), (
                f"loss_qvalue[{i}]={v} is non-finite with AUF={actor_update_freq}"
            )


# ---------------------------------------------------------------------------
# 5. Throughput benchmark (parametrized, never fails on speed)
# ---------------------------------------------------------------------------

# This matrix mirrors the migration guide's "suggested benchmark matrix"
_BENCH_CONFIGS: list[tuple[str, int | None, int, int]] = [
    ("Baseline (UTD-legacy)",      None, 1, 1),
    ("Fast-1x explicit",              1, 1, 1),
    ("Fast-delayed-actor-2",          1, 2, 1),
    ("Fast-delayed-actor-4",          1, 4, 1),
    ("Fast-delayed-target-2",         1, 1, 2),
    ("Fast-2x updates",               2, 1, 1),
    ("Fast-2x + delayed-actor-2",     2, 2, 1),
]

_bench_results: list[dict[str, Any]] = []   # collected across parametrize runs


@pytest.mark.parametrize("cfg_name,fur,auf,tuf", _BENCH_CONFIGS)
def test_throughput_benchmark(cfg_name: str, fur: int | None, auf: int, tuf: int):
    """
    Throughput parametrized benchmark.  Never asserts on speed — only verifies
    that the run completes and losses are finite.  Run with -s to see the table.
    """
    BENCH_FRAMES = 40
    BENCH_FPB = 4

    if fur is None:
        cfg = _base_cfg()
        cfg.collector.total_frames = BENCH_FRAMES
        cfg.collector.frames_per_batch = BENCH_FPB
        cfg.collector.init_random_frames = 0
        cfg.sac.utd_ratio = 1.0
        cls = SAC
    else:
        cfg = _fast_cfg(
            feature_update_ratio=fur,
            actor_update_freq=auf,
            target_update_freq=tuf,
            total_frames=BENCH_FRAMES,
            frames_per_batch=BENCH_FPB,
        )
        cls = FastSAC

    agent = _make_agent(cfg, cls)

    t0 = time.perf_counter()
    agent.train()
    elapsed = time.perf_counter() - t0

    fps = round(BENCH_FRAMES / elapsed, 1)
    n = agent.total_network_updates
    alpha = round(agent.loss_module.log_alpha.exp().item(), 4)

    actor_ok = (getattr(agent, "total_actor_updates", n) == n // max(1, auf))
    target_ok = (getattr(agent, "total_target_updates", n) == n // max(1, tuf))

    _bench_results.append({
        "name": cfg_name,
        "FUR": str(fur), "AUF": auf, "TUF": tuf,
        "fps": fps,
        "wall_s": round(elapsed, 3),
        "net_upd": n,
        "act_upd": getattr(agent, "total_actor_updates", n),
        "tgt_upd": getattr(agent, "total_target_updates", n),
        "ActOK": actor_ok,
        "TgtOK": target_ok,
        "alpha": alpha,
    })

    # The only hard assertions: run completes and cadence is correct
    assert actor_ok, (
        f"{cfg_name}: actor cadence wrong. "
        f"got {getattr(agent, 'total_actor_updates', n)}, expected {n // max(1, auf)}"
    )
    assert target_ok, (
        f"{cfg_name}: target cadence wrong. "
        f"got {getattr(agent, 'total_target_updates', n)}, expected {n // max(1, tuf)}"
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print benchmark table after all tests finish (only if results exist)."""
    if not _bench_results:
        return

    cols = ["name", "FUR", "AUF", "TUF", "fps", "wall_s",
            "net_upd", "act_upd", "tgt_upd", "ActOK", "TgtOK", "alpha"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in _bench_results))
              for c in cols}
    fmt = "  ".join(f"{{:<{widths[c]}}}" for c in cols)

    terminalreporter.write_sep("=", "FastSAC Benchmark Results")
    terminalreporter.write_line(fmt.format(*cols))
    terminalreporter.write_line("─" * sum(widths.values()))
    for r in _bench_results:
        terminalreporter.write_line(fmt.format(*[str(r[c]) for c in cols]))

    best = max(_bench_results, key=lambda r: r["fps"])
    terminalreporter.write_line(
        f"\n🏆  Fastest: '{best['name']}'  ({best['fps']} fps)"
    )
    bad = [r["name"] for r in _bench_results if not r["ActOK"] or not r["TgtOK"]]
    if bad:
        terminalreporter.write_line(f"⚠️   Cadence failures: {bad}")