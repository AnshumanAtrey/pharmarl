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
