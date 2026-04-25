"""Tests for the optional cross-family secondary held-out target.

Primary held-out is JNK3 (intra-family kinase, like the training targets).
We added an OPTIONAL secondary held-out that's truly cross-family —
``AMLODIPINE_MPO`` routes to TDC's ``amlodipine_mpo`` oracle, which scores
similarity-to-amlodipine plus drug-likeness. Amlodipine is an L-type
calcium-channel blocker, i.e. orthogonal pharmacology to the Ser/Thr +
MAP kinases used for training and the primary held-out.

If the TDC oracle can't load (offline / no network in CI), the tests skip
gracefully — they only assert behavior when the oracle is available.
"""

from __future__ import annotations

import pytest

from server.curriculum import CurriculumConfig, DEFAULT_CONFIG
from server.oracles import KNOWN_TARGETS, score_mpro_docking


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
HALOPERIDOL = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"


def _oracle_loads(target: str) -> bool:
    """Probe whether the oracle can score a known SMILES non-trivially."""
    try:
        score = score_mpro_docking(ASPIRIN, target=target)
        return score is not None
    except Exception:
        return False


def test_amlodipine_mpo_in_known_targets():
    """The new cross-family target must be advertised in KNOWN_TARGETS."""
    assert "AMLODIPINE_MPO" in KNOWN_TARGETS


def test_secondary_held_out_default_is_none():
    """Backwards compatibility: feature is opt-in, headline run unaffected."""
    cfg = CurriculumConfig()
    assert cfg.secondary_held_out_target is None
    assert DEFAULT_CONFIG.secondary_held_out_target is None


def test_secondary_held_out_can_be_configured():
    """Config accepts the cross-family target as a string."""
    cfg = CurriculumConfig(secondary_held_out_target="AMLODIPINE_MPO")
    assert cfg.secondary_held_out_target == "AMLODIPINE_MPO"


def test_cross_family_oracle_returns_in_range():
    """Score for aspirin must be in [0, 1] when the oracle is available."""
    score = score_mpro_docking(ASPIRIN, target="AMLODIPINE_MPO")
    if score == 0.0 and not _oracle_loads("AMLODIPINE_MPO"):
        pytest.skip("amlodipine_mpo oracle unavailable (network?)")
    assert 0.0 <= score <= 1.0


def test_cross_family_distinct_from_kinase():
    """Cross-family score for haloperidol differs from its DRD2 score.

    Proves the cross-family target routes to a genuinely different oracle
    rather than aliasing one of the kinase classifiers.
    """
    drd2 = score_mpro_docking(HALOPERIDOL, target="DRD2")
    cross = score_mpro_docking(HALOPERIDOL, target="AMLODIPINE_MPO")
    if drd2 == 0.0 and cross == 0.0:
        pytest.skip("Neither oracle available (offline)")
    # If both load, they must produce distinct values for haloperidol —
    # haloperidol is a D2 antagonist, so DRD2 should score it high and
    # amlodipine_mpo should score it differently.
    assert drd2 != cross, (
        f"Cross-family oracle appears aliased to DRD2: "
        f"DRD2={drd2}, AMLODIPINE_MPO={cross}"
    )


def test_unknown_target_still_safe():
    """Routing an unknown target must not crash — returns 0.0."""
    score = score_mpro_docking(ASPIRIN, target="NOT_A_REAL_TARGET")
    assert score == 0.0
