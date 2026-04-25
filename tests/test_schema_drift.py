"""Schema-drift mechanic tests (Patronus AI sub-theme).

The schema-drift feature lets reward weights change mid-episode, modeling
real medicinal-chemistry workflows where constraints shift mid-development.

Default behavior (schema_drift_enabled=False) MUST be identical to the legacy
implementation — these tests pin both the new behavior AND the
backwards-compatibility guarantee.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from server.curriculum import (
    DEFAULT_CONFIG,
    CurriculumConfig,
    pick_drift_profile,
    weights_for,
)
from server.grader import (
    W_DOCKING,
    W_QED,
    W_SA,
    W_TOX,
    terminal_reward,
)


# ─── 1. Default-off invariant ──────────────────────────────────────────────


def test_drift_disabled_by_default_means_static_weights() -> None:
    """When schema_drift_enabled=False, weights match the existing static defaults.

    This is the firewall around Sahil's headline run — flipping any other knob
    in CurriculumConfig must not perturb the legacy reward formula.
    """
    assert DEFAULT_CONFIG.schema_drift_enabled is False
    # pick_drift_profile must always return "static" when the master flag is OFF,
    # regardless of how many times we sample.
    import random
    rng = random.Random(0)
    for _ in range(50):
        assert pick_drift_profile(rng, DEFAULT_CONFIG) == "static"

    # Static profile = static weights at every step.
    for step in range(0, 30):
        w = weights_for("static", step, DEFAULT_CONFIG.drift_step, DEFAULT_CONFIG)
        assert w == DEFAULT_CONFIG.weights_static
        # And the static weights themselves equal the legacy module constants.
        assert w == (W_DOCKING, W_QED, W_SA, W_TOX)


# ─── 2. Drift fires at the configured step ─────────────────────────────────


def test_early_admet_drift_changes_weights_at_drift_step() -> None:
    """early_admet: pre-weights before drift_step, post-weights from drift_step on."""
    cfg = DEFAULT_CONFIG
    drift_step = cfg.drift_step

    # Just before the drift fires — pre weights.
    for s in range(0, drift_step):
        w = weights_for("early_admet", s, drift_step, cfg)
        assert w == cfg.weights_early_admet_pre, (
            f"step {s} (< drift_step={drift_step}) should give pre-weights"
        )

    # On and after the drift step — post weights.
    for s in range(drift_step, drift_step + 5):
        w = weights_for("early_admet", s, drift_step, cfg)
        assert w == cfg.weights_early_admet_post, (
            f"step {s} (>= drift_step={drift_step}) should give post-weights"
        )

    # Sanity: pre and post are actually different — otherwise the test is hollow.
    assert cfg.weights_early_admet_pre != cfg.weights_early_admet_post


def test_late_potency_drift_changes_weights_at_drift_step() -> None:
    """late_potency mirrors early_admet — the pre/post split fires at drift_step."""
    cfg = DEFAULT_CONFIG
    drift_step = cfg.drift_step

    assert weights_for("late_potency", 0, drift_step, cfg) == cfg.weights_late_potency_pre
    assert weights_for("late_potency", drift_step - 1, drift_step, cfg) == cfg.weights_late_potency_pre
    assert weights_for("late_potency", drift_step, drift_step, cfg) == cfg.weights_late_potency_post
    assert weights_for("late_potency", drift_step + 1, drift_step, cfg) == cfg.weights_late_potency_post

    assert cfg.weights_late_potency_pre != cfg.weights_late_potency_post


def test_unknown_profile_falls_back_to_static() -> None:
    """Defensive: a typo'd profile must not silently produce weird weights."""
    cfg = DEFAULT_CONFIG
    assert weights_for("does_not_exist", 0, cfg.drift_step, cfg) == cfg.weights_static
    assert weights_for("does_not_exist", 99, cfg.drift_step, cfg) == cfg.weights_static


# ─── 3. terminal_reward honors dynamic weights ─────────────────────────────


def test_terminal_reward_accepts_dynamic_weights() -> None:
    """terminal_reward(weights=...) should produce a different composite than the default."""
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin

    default_result = terminal_reward(
        smiles,
        components_active=("qed", "docking", "sa", "toxicity"),
        target="DRD2",
    )

    # Heavily skewed weights — almost everything on QED. Composite must move.
    skewed = (0.05, 0.85, 0.05, 0.05)
    skewed_result = terminal_reward(
        smiles,
        components_active=("qed", "docking", "sa", "toxicity"),
        target="DRD2",
        weights=skewed,
    )

    # The two composites should be measurably different — otherwise the
    # weights argument is being ignored.
    assert abs(default_result.composite - skewed_result.composite) > 1e-6, (
        f"weights override had no effect: default={default_result.composite:.4f} "
        f"skewed={skewed_result.composite:.4f}"
    )


def test_terminal_reward_weights_none_is_backwards_compat() -> None:
    """Calling terminal_reward without ``weights=`` must behave exactly like before."""
    smiles = "CC(=O)Oc1ccccc1C(=O)O"
    legacy = terminal_reward(smiles, components_active=("qed", "docking", "sa", "toxicity"), target="DRD2")
    explicit_default = terminal_reward(
        smiles,
        components_active=("qed", "docking", "sa", "toxicity"),
        target="DRD2",
        weights=(W_DOCKING, W_QED, W_SA, W_TOX),
    )
    assert abs(legacy.composite - explicit_default.composite) < 1e-9
    assert abs(legacy.reward - explicit_default.reward) < 1e-9


# ─── 4. End-to-end via the env (integration smoke test) ────────────────────


def test_env_reset_default_config_uses_static_profile() -> None:
    """An env constructed with DEFAULT_CONFIG must always run the 'static' profile."""
    from server.drug_discovery_environment import DrugDiscoveryEnvironment

    env = DrugDiscoveryEnvironment(seed=7)
    # Sample many resets — every one should be static when the flag is off.
    for _ in range(20):
        env.reset(difficulty="trivial")
        assert env.state.drift_profile == "static"


def test_env_reset_with_drift_enabled_can_pick_non_static() -> None:
    """When schema_drift_enabled=True, the env eventually picks a non-static profile."""
    from server.drug_discovery_environment import DrugDiscoveryEnvironment

    cfg = replace(DEFAULT_CONFIG, schema_drift_enabled=True)
    env = DrugDiscoveryEnvironment(seed=123, config=cfg)

    seen = set()
    for _ in range(50):
        env.reset(difficulty="trivial")
        seen.add(env.state.drift_profile)
    # With 50 samples over a 3-element pool, we should hit at least 2 of them.
    assert len(seen) >= 2, f"only saw drift profiles {seen} — sampler may be stuck"
    # All sampled profiles must be in the configured pool.
    assert seen.issubset(set(cfg.drift_profiles))


def test_env_observation_includes_drift_metadata() -> None:
    """The observation must surface drift profile / weights / active constraints."""
    from server.drug_discovery_environment import DrugDiscoveryEnvironment
    from models import MoleculeAction

    cfg = replace(DEFAULT_CONFIG, schema_drift_enabled=True)
    env = DrugDiscoveryEnvironment(seed=42, config=cfg)
    obs = env.reset(difficulty="trivial", drift_profile="early_admet")

    assert obs.metadata["drift_profile"] == "early_admet"
    assert obs.metadata["drift_step"] == cfg.drift_step
    assert isinstance(obs.metadata["weights"], list) and len(obs.metadata["weights"]) == 4
    # active_constraints reflects the pre-weights (binding+drug-like only).
    assert "docking" in obs.active_constraints
    assert "qed" in obs.active_constraints
    assert "sa" not in obs.active_constraints
    assert "toxicity" not in obs.active_constraints


def test_env_drift_warning_fires_on_drift_step() -> None:
    """A drift_warning string must appear on the step where weights flip."""
    from server.drug_discovery_environment import DrugDiscoveryEnvironment
    from models import MoleculeAction

    # Use a small drift_step so the test is fast and deterministic.
    cfg = replace(
        DEFAULT_CONFIG,
        schema_drift_enabled=True,
        drift_step=2,
        # Generous max_steps so we can reach the drift step on the trivial tier.
        trivial_max_steps=30,
    )
    env = DrugDiscoveryEnvironment(seed=99, config=cfg)
    env.reset(difficulty="trivial", drift_profile="early_admet")

    saw_warning = False
    for i in range(cfg.drift_step + 2):
        obs = env.step(MoleculeAction(action_type="ADD_FRAGMENT", fragment="C", position=0))
        if obs.drift_warning:
            saw_warning = True
            # Warning should reference the drift step explicitly.
            assert "Schema change" in obs.drift_warning
            assert f"step {cfg.drift_step}" in obs.drift_warning
            break
        if obs.done:
            break
    assert saw_warning, "drift_warning never appeared even though drift_step was crossed"
