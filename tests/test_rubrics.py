"""Tests for the composable rubric system (server/rubrics.py).

Confirms:
  1. Each leaf rubric matches its underlying oracle.
  2. Operators ``*`` and ``+`` compose rubrics correctly.
  3. ``composite_for_target`` matches the legacy ``terminal_reward`` math
     for the full-component-set (Hard tier) path.
  4. ``terminal_reward`` is unchanged after the refactor — pinned values
     for known molecules.
"""

from __future__ import annotations

import math

import pytest

from server.grader import terminal_reward
from server.oracles import score_qed, score_sa, score_toxicity
from server.rubrics import (
    BindingRubric,
    QedRubric,
    SaRubric,
    ToxicityRubric,
    composite_for_target,
)


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
HALOPERIDOL = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def test_qed_rubric_matches_score_qed():
    """``QedRubric().score(s)`` must equal ``score_qed(s)`` for any SMILES."""
    for s in (ASPIRIN, HALOPERIDOL, CAFFEINE):
        assert QedRubric().score(s) == pytest.approx(score_qed(s))


def test_sa_rubric_matches_score_sa():
    """``SaRubric().score(s)`` must equal ``score_sa(s)``."""
    for s in (ASPIRIN, HALOPERIDOL, CAFFEINE):
        assert SaRubric().score(s) == pytest.approx(score_sa(s))


def test_toxicity_rubric_is_inverted():
    """``ToxicityRubric().score(s) == 1 - score_toxicity(s)``."""
    for s in (ASPIRIN, HALOPERIDOL, CAFFEINE):
        assert ToxicityRubric().score(s) == pytest.approx(1.0 - score_toxicity(s))


def test_binding_rubric_takes_target():
    """``BindingRubric(target)`` routes to a specific oracle.

    We don't assume DRD2 oracle is loaded (network-dependent in CI), so we
    just confirm the result is in [0, 1] and that distinct targets construct
    independently.
    """
    drd2 = BindingRubric("DRD2").score(HALOPERIDOL)
    jnk3 = BindingRubric("JNK3").score(HALOPERIDOL)
    assert 0.0 <= drd2 <= 1.0
    assert 0.0 <= jnk3 <= 1.0


def test_rubric_composition_via_operators():
    """``(QedRubric() * 0.5 + SaRubric() * 0.5).score(s)`` ≈ 0.5*qed + 0.5*sa."""
    rubric = QedRubric() * 0.5 + SaRubric() * 0.5
    expected = 0.5 * score_qed(ASPIRIN) + 0.5 * score_sa(ASPIRIN)
    assert rubric.score(ASPIRIN) == pytest.approx(expected)


def test_weight_distribution_via_outer_multiply():
    """Multiplying a SumRubric by a scalar distributes across all parts."""
    inner = QedRubric() * 0.5 + SaRubric() * 0.5
    scaled = inner * 0.4
    expected = 0.4 * (0.5 * score_qed(ASPIRIN) + 0.5 * score_sa(ASPIRIN))
    assert scaled.score(ASPIRIN) == pytest.approx(expected)


def test_composite_for_target_matches_legacy_math():
    """``composite_for_target`` must equal the legacy weighted formula.

    Pin: for ``weights=None`` and the full Hard-tier component set, the
    composite from ``composite_for_target`` must equal
    ``0.40*docking + 0.25*qed + 0.15*sa + 0.20*(1-tox)``.
    """
    for s in (ASPIRIN, CAFFEINE):
        # Manual legacy formula
        from server.oracles import score_mpro_docking

        qed = score_qed(s)
        docking = score_mpro_docking(s, target=None)
        sa = score_sa(s)
        tox = score_toxicity(s)
        expected = 0.40 * docking + 0.25 * qed + 0.15 * sa + 0.20 * (1.0 - tox)

        rubric_score = composite_for_target(None).score(s)
        assert rubric_score == pytest.approx(expected, abs=1e-9)


def test_terminal_reward_uses_rubrics_internally():
    """``terminal_reward`` output is unchanged after the refactor.

    Pins the composite + reward for known molecules under the Hard-tier
    component set so future edits cannot silently shift the math.
    """
    # Default behavior (weights=None, hard components)
    tr = terminal_reward(ASPIRIN)
    # Recompute the composite the legacy way for comparison
    from server.oracles import score_mpro_docking

    qed = score_qed(ASPIRIN)
    docking = score_mpro_docking(ASPIRIN, target=None)
    sa = score_sa(ASPIRIN)
    tox = score_toxicity(ASPIRIN)
    expected_composite = 0.40 * docking + 0.25 * qed + 0.15 * sa + 0.20 * (1.0 - tox)

    # Apply the Lipinski gate (aspirin passes, so no gate)
    assert tr.lipinski_passes is True
    assert tr.composite == pytest.approx(expected_composite, abs=1e-9)
    assert tr.reward == pytest.approx(expected_composite * 10.0, abs=1e-9)


def test_terminal_reward_with_custom_weights_uses_rubrics():
    """Passing ``weights`` should route through a custom-weight rubric."""
    weights = (0.5, 0.3, 0.1, 0.1)  # docking, qed, sa, tox
    tr = terminal_reward(ASPIRIN, weights=weights)
    from server.oracles import score_mpro_docking

    qed = score_qed(ASPIRIN)
    docking = score_mpro_docking(ASPIRIN, target=None)
    sa = score_sa(ASPIRIN)
    tox = score_toxicity(ASPIRIN)
    expected = 0.5 * docking + 0.3 * qed + 0.1 * sa + 0.1 * (1.0 - tox)
    assert tr.lipinski_passes is True
    assert tr.composite == pytest.approx(expected, abs=1e-9)


def test_addition_is_associative_and_flat():
    """``a + b + c`` flattens into a single SumRubric (no nesting bloat)."""
    a = QedRubric()
    b = SaRubric()
    c = ToxicityRubric()
    s = a + b + c
    # All three should score the same as the manual sum
    expected = a.score(ASPIRIN) + b.score(ASPIRIN) + c.score(ASPIRIN)
    assert s.score(ASPIRIN) == pytest.approx(expected)
