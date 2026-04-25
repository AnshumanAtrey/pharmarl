"""Synthetic Accessibility score (Ertl & Schuffenhauer 2009).

Estimates how easy a molecule is to synthesize. 1 = very easy, 10 = very hard.
We invert and normalize to 0-1 so higher is better, matching reward convention.

We try TDC's Oracle('SA') first (battle-tested wrapper). If unavailable,
fall back to RDKit's sascorer if present, else return a neutral 0.5.
"""

from __future__ import annotations

from rdkit import Chem

_TDC_SA_ORACLE = None
_RDKIT_SA_FN = None


def _lazy_init() -> None:
    """Load whichever SA scorer is available. Cached after first call."""
    global _TDC_SA_ORACLE, _RDKIT_SA_FN
    if _TDC_SA_ORACLE is not None or _RDKIT_SA_FN is not None:
        return
    # Try TDC first
    try:
        from tdc import Oracle  # type: ignore
        _TDC_SA_ORACLE = Oracle(name="SA")
        return
    except Exception:
        pass
    # Try RDKit's sascorer (ships with RDKit Contrib)
    try:
        from rdkit.Chem import RDConfig  # type: ignore
        import os
        import sys
        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        import sascorer  # type: ignore
        _RDKIT_SA_FN = sascorer.calculateScore
    except Exception:
        _RDKIT_SA_FN = None


def score_sa(smiles: str) -> float:
    """Returns 0-1 (higher is more synthesizable). 0.5 if unavailable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    _lazy_init()

    raw_sa: float | None = None
    if _TDC_SA_ORACLE is not None:
        try:
            raw_sa = float(_TDC_SA_ORACLE(smiles))
        except Exception:
            raw_sa = None
    if raw_sa is None and _RDKIT_SA_FN is not None:
        try:
            raw_sa = float(_RDKIT_SA_FN(mol))
        except Exception:
            raw_sa = None
    if raw_sa is None:
        return 0.5  # neutral when no scorer available

    # SA is 1..10 (low = easy). Invert + normalize to 0..1.
    raw_sa = max(1.0, min(10.0, raw_sa))
    return float((10.0 - raw_sa) / 9.0)
