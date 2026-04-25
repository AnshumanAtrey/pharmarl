"""Lipinski + SMILES/SELFIES round-trip tests."""

from __future__ import annotations

from server.molecule_engine.validation import (
    canonicalize_smiles,
    check_lipinski,
    is_valid_molecule,
    selfies_to_smiles,
    smiles_to_selfies,
)


def test_aspirin_passes_lipinski() -> None:
    res = check_lipinski("CC(=O)Oc1ccccc1C(=O)O")
    assert res is not None
    assert res.passes
    assert res.violations == 0
    assert res.mw < 500


def test_invalid_smiles_returns_none() -> None:
    assert check_lipinski("invalid_!@#") is None
    assert canonicalize_smiles("invalid_!@#") is None
    assert not is_valid_molecule("invalid_!@#")


def test_canonicalize_normalizes() -> None:
    assert canonicalize_smiles("c1ccccc1") == canonicalize_smiles("C1=CC=CC=C1")


def test_selfies_roundtrip() -> None:
    smi = "CC(=O)Oc1ccccc1C(=O)O"
    sf = smiles_to_selfies(smi)
    assert sf is not None and len(sf) > 0
    back = selfies_to_smiles(sf)
    assert back is not None
    # Round-trip should yield a valid molecule (canonical form may differ)
    assert is_valid_molecule(back)


def test_oversize_molecule_fails_lipinski() -> None:
    # Long aliphatic chain → MW > 500
    big = "C" * 50
    res = check_lipinski(big)
    if res is not None:
        assert res.violations >= 1
