"""Toxicity oracle — CYP3A4 inhibition (TDC ADMET classifier).

CYP3A4 metabolizes ~50% of clinical drugs. Inhibitors cause drug-drug
interactions and are flagged in ADMET screens. Returns 0-1 probability of
inhibition; we invert in the grader so lower-toxicity scores higher reward.

Falls back to a simple Lipinski-violation heuristic if TDC oracle unavailable.
"""

from __future__ import annotations

from rdkit import Chem

from ..molecule_engine.validation import check_lipinski

_TDC_TOX_ORACLE = None
_TRIED_TDC_INIT = False


def _lazy_init() -> None:
    global _TDC_TOX_ORACLE, _TRIED_TDC_INIT
    if _TRIED_TDC_INIT:
        return
    _TRIED_TDC_INIT = True
    candidates = ["CYP3A4_Veith", "CYP3A4", "CYP2C9_Veith", "hERG"]
    try:
        from tdc import Oracle  # type: ignore
        for name in candidates:
            try:
                _TDC_TOX_ORACLE = Oracle(name=name)
                return
            except Exception:
                continue
    except Exception:
        _TDC_TOX_ORACLE = None


def score_toxicity(smiles: str) -> float:
    """Returns 0-1 toxicity probability (higher = more toxic).

    Reward function will subtract a fraction of this, so lower is better.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 1.0  # max toxicity penalty for invalid molecules

    _lazy_init()
    if _TDC_TOX_ORACLE is not None:
        try:
            return float(_TDC_TOX_ORACLE(smiles))
        except Exception:
            pass

    # Fallback: Lipinski violations as a crude toxicity proxy
    res = check_lipinski(smiles)
    if res is None:
        return 1.0
    return float(res.violations) / 4.0
