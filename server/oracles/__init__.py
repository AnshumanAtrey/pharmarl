"""Oracles — reward signal sources.

Each oracle takes a SMILES string and returns a float score. They are
called by the grader and combined into a composite terminal reward.

  - docking_mpro: TDC SARS-CoV-2 Mpro docking (or DRD2 fallback)
  - qed:          RDKit drug-likeness (0-1)
  - sa:           RDKit synthetic accessibility (1-10, lower is better)
  - toxicity:     TDC CYP3A4 inhibition (0-1, lower is better)
"""

from .docking_mpro import score_mpro_docking
from .qed import score_qed
from .sa import score_sa
from .toxicity import score_toxicity

__all__ = [
    "score_mpro_docking",
    "score_qed",
    "score_sa",
    "score_toxicity",
]
