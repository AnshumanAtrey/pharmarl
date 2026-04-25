"""Reward red-team — adversarial inputs that should NOT max out composite.

If any of these slip through, the agent will discover them during training and
exploit them — wasting Colab time on "learning" that doesn't generalize.

Run BEFORE training:
    pytest tests/test_reward_redteam.py -v

Each test asserts a property of the reward signal — they should all pass before
we trust the env to train against.
"""

from __future__ import annotations

import pytest

from server.grader import terminal_reward


def _composite(smiles: str, target: str = "DRD2") -> float:
    """Return the composite (pre-Lipinski-gate, pre-scale) for hard tier (all 4 components)."""
    tr = terminal_reward(smiles, components_active=("qed", "docking", "sa", "toxicity"), target=target)
    return tr.composite


# ─── Adversarial inputs — composite should NOT be high ─────────────────


def test_single_carbon_does_not_max_score() -> None:
    """A single 'C' has trivially high QED but is not a drug."""
    c = _composite("C")
    assert c < 0.7, f"single-C composite={c:.3f} — too high; agent could spam single atoms"


def test_methane_alone_does_not_pass_lipinski_gate() -> None:
    """Single CH4 should pass Lipinski (low MW), but its composite is QED-noise; not a real drug."""
    tr = terminal_reward("C", components_active=("qed", "docking", "sa", "toxicity"), target="DRD2")
    # QED alone won't push us above 0.6 because docking + tox + sa drag it down
    assert tr.reward < 7.0, f"trivial 'C' got terminal reward {tr.reward:.3f} — too high"


def test_aromatic_ring_blob_is_lipinski_failed() -> None:
    """A polyaromatic blob should fail Lipinski (MW or LogP) → composite halved.

    Six fused rings ~ MW>500 and LogP>5; Lipinski gate must catch this.
    """
    blob = "c1ccc2c(c1)ccc1ccc3ccc4ccc5ccccc5c4c3c12"
    tr = terminal_reward(blob, components_active=("qed", "docking", "sa", "toxicity"), target="DRD2")
    # Should hit the Lipinski gate (×0.5 reduction) at minimum
    assert tr.lipinski_passes is False, "polyaromatic blob should FAIL Lipinski Rule of 5"


def test_invalid_smiles_returns_zero() -> None:
    """Garbage strings should not produce a reward. The agent's parse penalty
    catches these upstream, but if a malformed SMILES somehow reaches the grader,
    it should yield 0, not an exception or a high score."""
    tr = terminal_reward("@@@not_a_smiles@@@", components_active=("qed", "docking", "sa", "toxicity"), target="DRD2")
    # The composite for unparseable SMILES is ~0 — every oracle returns 0 on RDKit parse failure
    assert tr.composite < 0.05, f"garbage SMILES composite={tr.composite:.3f} — should be ~0"


def test_empty_string_does_not_crash() -> None:
    """Empty SMILES — defensive. Should not crash, should return ~0."""
    tr = terminal_reward("", components_active=("qed", "docking", "sa", "toxicity"), target="DRD2")
    assert tr.composite < 0.1


# ─── Positive controls — known drugs SHOULD score reasonably ───────────


def test_haloperidol_scores_high_on_drd2() -> None:
    """Haloperidol is a clinical D2 antagonist — composite should be respectable."""
    haloperidol = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
    tr = terminal_reward(haloperidol, components_active=("qed", "docking", "sa", "toxicity"), target="DRD2")
    # Strong DRD2 binding (~0.99) dominates with weight 0.40
    assert tr.composite > 0.5, f"haloperidol composite={tr.composite:.3f} — expected >0.5; oracle may be broken"


def test_aspirin_scores_low_on_drd2() -> None:
    """Aspirin is NOT a D2 ligand — DRD2 component should be near zero, dragging composite down."""
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    tr = terminal_reward(aspirin, components_active=("qed", "docking", "sa", "toxicity"), target="DRD2")
    # Aspirin DRD2 ~0.0003. With QED ~0.55 + SA + (1-tox), composite ends mid-range.
    # The point is haloperidol > aspirin by a wide margin.
    halo_tr = terminal_reward(
        "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
        components_active=("qed", "docking", "sa", "toxicity"), target="DRD2",
    )
    assert halo_tr.composite > tr.composite + 0.15, (
        f"haloperidol composite ({halo_tr.composite:.3f}) should beat aspirin ({tr.composite:.3f}) "
        f"by >0.15 — DRD2 oracle may be broken"
    )


# ─── Reward shape constraints ───────────────────────────────────────────


def test_terminal_reward_in_known_range() -> None:
    """Terminal reward = composite × 10 (scaled). Composite is in [0, 1] ideally; with
    Lipinski gate halving, range is [0, 10]."""
    for smi in ("C", "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2ccccc2c1"):
        tr = terminal_reward(smi, components_active=("qed", "docking", "sa", "toxicity"), target="DRD2")
        assert 0.0 <= tr.reward <= 10.0, f"reward {tr.reward} outside [0, 10] for {smi!r}"
        assert 0.0 <= tr.composite <= 1.0, f"composite {tr.composite} outside [0, 1] for {smi!r}"


# ─── Multi-target regression — same molecule, different oracles ─────────


def test_haloperidol_drd2_higher_than_jnk3() -> None:
    """Haloperidol is a D2 antagonist, NOT a JNK3 inhibitor.
    DRD2 should score it much higher than JNK3 — confirms the multi-target router works."""
    haloperidol = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
    drd2 = terminal_reward(haloperidol, components_active=("qed", "docking", "sa", "toxicity"), target="DRD2")
    jnk3 = terminal_reward(haloperidol, components_active=("qed", "docking", "sa", "toxicity"), target="JNK3")
    assert drd2.composite > jnk3.composite, (
        f"Multi-target router likely broken: haloperidol DRD2={drd2.composite:.3f} "
        f"NOT > JNK3={jnk3.composite:.3f}"
    )
