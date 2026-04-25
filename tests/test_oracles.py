"""Oracle smoke tests — confirm scoring functions return expected ranges."""

from __future__ import annotations

import pytest


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
INVALID = "this_is_not_a_smiles_string_!@#"


def test_qed_aspirin_in_range() -> None:
    from server.oracles.qed import score_qed
    score = score_qed(ASPIRIN)
    assert 0.0 <= score <= 1.0
    assert score > 0.3  # aspirin QED ≈ 0.55, well above noise


def test_qed_invalid_returns_zero() -> None:
    from server.oracles.qed import score_qed
    assert score_qed(INVALID) == 0.0


def test_sa_returns_in_range() -> None:
    from server.oracles.sa import score_sa
    s = score_sa(ASPIRIN)
    assert 0.0 <= s <= 1.0


def test_toxicity_returns_in_range() -> None:
    from server.oracles.toxicity import score_toxicity
    t = score_toxicity(ASPIRIN)
    assert 0.0 <= t <= 1.0


def test_docking_returns_in_range() -> None:
    from server.oracles.docking_mpro import score_mpro_docking
    d = score_mpro_docking(ASPIRIN)
    assert 0.0 <= d <= 1.0


def test_invalid_smiles_safe_for_all_oracles() -> None:
    from server.oracles import (
        score_mpro_docking,
        score_qed,
        score_sa,
        score_toxicity,
    )
    # None should crash on invalid SMILES
    for fn in (score_qed, score_sa, score_toxicity, score_mpro_docking):
        out = fn(INVALID)
        assert isinstance(out, float)


def test_multi_target_oracles_return_in_range() -> None:
    """The kinase trio (used for the held-out generalization test) must each
    return a number in [0, 1] without falling back to the default oracle.
    The optional cross-family ``AMLODIPINE_MPO`` target is also exposed but
    only as an opt-in secondary held-out — see tests/test_cross_family.py."""
    from server.oracles import KNOWN_TARGETS, score_mpro_docking
    # Primary kinase targets must all be present.
    for kinase in ("DRD2", "GSK3B", "JNK3"):
        assert kinase in KNOWN_TARGETS
        score = score_mpro_docking(ASPIRIN, target=kinase)
        assert 0.0 <= score <= 1.0, f"target={kinase!r} returned out-of-range {score}"


def test_unknown_target_returns_zero() -> None:
    """Unknown targets shouldn't crash — they fall back to 0.0 with a warning."""
    from server.oracles import score_mpro_docking
    assert score_mpro_docking(ASPIRIN, target="not_a_real_target") == 0.0


def test_drd2_haloperidol_scores_high() -> None:
    """Sanity: haloperidol (a known D2 antagonist drug) should score >0.9 on DRD2.
    Catches model-cache corruption before training burns time."""
    from server.oracles import score_mpro_docking
    haloperidol = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
    score = score_mpro_docking(haloperidol, target="DRD2")
    assert score > 0.9, f"DRD2(haloperidol) = {score} — expected > 0.9; oracle likely corrupted"
