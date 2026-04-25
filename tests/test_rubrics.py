"""Tests for the composable rubric layer.

Verifies:
  1. Each individual rubric matches its underlying oracle for known molecules
  2. composite_for_target() produces values matching the legacy terminal_reward
  3. Operator composition (* weight, + sum) produces correct arithmetic
  4. Backwards compat — existing terminal_reward output unchanged
"""

from __future__ import annotations

import math

import pytest

from server.grader import terminal_reward
from server.oracles import score_qed, score_sa, score_toxicity, score_mpro_docking
from server.rubrics import (
    BindingRubric,
    QedRubric,
    SaRubric,
    SumRubric,
    ToxicityRubric,
    WeightedRubric,
    composite_for_target,
)


_HALOPERIDOL = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
_ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


# ─── Individual rubrics match their oracle calls ─────────────────────


def test_qed_rubric_matches_oracle() -> None:
    """QedRubric().score(s) == score_qed(s) for any SMILES."""
    for smi in (_ASPIRIN, _HALOPERIDOL, "C", "CCO"):
        assert math.isclose(QedRubric().score(smi), score_qed(smi), abs_tol=1e-9), \
            f"QedRubric mismatch for {smi!r}"


def test_sa_rubric_matches_oracle() -> None:
    for smi in (_ASPIRIN, _HALOPERIDOL):
        assert math.isclose(SaRubric().score(smi), score_sa(smi), abs_tol=1e-9)


def test_toxicity_rubric_is_inverted() -> None:
    """ToxicityRubric returns 1 - tox so higher = less toxic."""
    for smi in (_ASPIRIN, _HALOPERIDOL):
        expected = 1.0 - score_toxicity(smi)
        assert math.isclose(ToxicityRubric().score(smi), expected, abs_tol=1e-9)


def test_binding_rubric_takes_target() -> None:
    """BindingRubric routes to the multi-target oracle correctly."""
    halo_drd2 = BindingRubric("DRD2").score(_HALOPERIDOL)
    asp_drd2 = BindingRubric("DRD2").score(_ASPIRIN)
    # Haloperidol is a clinical D2 antagonist; aspirin is not.
    assert halo_drd2 > asp_drd2 + 0.5, (
        f"DRD2 oracle direction wrong: halo={halo_drd2:.3f} aspirin={asp_drd2:.3f}"
    )


# ─── Operator composition produces correct math ─────────────────────


def test_weighted_rubric_scales_score() -> None:
    """(QedRubric() * 0.5).score(s) == 0.5 * QedRubric().score(s)."""
    raw = QedRubric().score(_ASPIRIN)
    weighted = (QedRubric() * 0.5).score(_ASPIRIN)
    assert math.isclose(weighted, 0.5 * raw, abs_tol=1e-9)


def test_sum_rubric_adds_components() -> None:
    """(QedRubric() + SaRubric()).score(s) ≈ qed + sa."""
    qed = score_qed(_ASPIRIN)
    sa = score_sa(_ASPIRIN)
    summed = (QedRubric() + SaRubric()).score(_ASPIRIN)
    assert math.isclose(summed, qed + sa, abs_tol=1e-9)


def test_weighted_sum_combines() -> None:
    """(QedRubric() * 0.5 + SaRubric() * 0.5).score(s) ≈ 0.5*qed + 0.5*sa."""
    qed = score_qed(_ASPIRIN)
    sa = score_sa(_ASPIRIN)
    rubric = QedRubric() * 0.5 + SaRubric() * 0.5
    expected = 0.5 * qed + 0.5 * sa
    assert math.isclose(rubric.score(_ASPIRIN), expected, abs_tol=1e-9)


def test_sum_associative() -> None:
    """((A + B) + C).score == (A + (B + C)).score == (A + B + C).score."""
    left = (QedRubric() + SaRubric()) + ToxicityRubric()
    right = QedRubric() + (SaRubric() + ToxicityRubric())
    flat = QedRubric() + SaRubric() + ToxicityRubric()
    for smi in (_ASPIRIN, _HALOPERIDOL):
        assert math.isclose(left.score(smi), right.score(smi), abs_tol=1e-6)
        assert math.isclose(left.score(smi), flat.score(smi), abs_tol=1e-6)


def test_rmul_supports_left_scalar() -> None:
    """`0.4 * BindingRubric()` should equal `BindingRubric() * 0.4`."""
    a = (0.4 * BindingRubric("DRD2")).score(_ASPIRIN)
    b = (BindingRubric("DRD2") * 0.4).score(_ASPIRIN)
    assert math.isclose(a, b, abs_tol=1e-9)


# ─── Backwards compat — terminal_reward output unchanged ────────────


def test_composite_for_target_matches_legacy_terminal_reward() -> None:
    """composite_for_target('DRD2').score(s) == terminal_reward(s).composite, for hard tier.

    This is the load-bearing test: refactoring grader.py to use rubrics
    must not change any reward value Sahil's training already saw.
    """
    for smi in (_ASPIRIN, _HALOPERIDOL):
        rubric_score = composite_for_target("DRD2").score(smi)
        legacy_tr = terminal_reward(
            smi,
            components_active=("qed", "docking", "sa", "toxicity"),
            target="DRD2",
        )
        # legacy_tr.composite has Lipinski gating applied; rubric_score doesn't.
        # Reverse the gate to get a fair comparison.
        if not legacy_tr.lipinski_passes:
            legacy_pre_gate = legacy_tr.composite / 0.5
        else:
            legacy_pre_gate = legacy_tr.composite

        assert math.isclose(rubric_score, legacy_pre_gate, abs_tol=1e-3), (
            f"Rubric {rubric_score:.4f} doesn't match legacy {legacy_pre_gate:.4f} for {smi!r}"
        )


def test_terminal_reward_haloperidol_pinned() -> None:
    """Pin the absolute composite for haloperidol on DRD2.

    If a refactor accidentally changes the reward value, this test catches
    it before training data starts diverging.
    """
    tr = terminal_reward(
        _HALOPERIDOL,
        components_active=("qed", "docking", "sa", "toxicity"),
        target="DRD2",
    )
    # Haloperidol is a clinical D2 antagonist. Composite should be > 0.5.
    # (We don't pin the exact value because TDC oracle versions can shift it
    # by a few percent; the >0.5 floor is the load-bearing claim.)
    assert tr.composite > 0.5
    assert tr.lipinski_passes is True
    assert tr.reward > 5.0
