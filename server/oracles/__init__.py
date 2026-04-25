"""Oracles — reward signal sources.

Each oracle takes a SMILES string and returns a float score. They are
called by the grader and combined into a composite terminal reward.

  - docking_mpro: binding score; default = TDC DRD2 classifier (canonical
                  MOSES/GuacaMol benchmark). With PHARMARL_ENABLE_DOCKING=1
                  and pyscreener+Vina installed, falls forward to docking
                  on NSP15/EGFR/ABL/BACE1.
  - qed:          RDKit drug-likeness (0-1)
  - sa:           RDKit synthetic accessibility (1-10, lower is better)
  - toxicity:     TDC CYP3A4 inhibition (0-1, lower is better)
"""

from .docking_mpro import (
    get_active_oracle_name,
    get_active_target_name,
    score_mpro_docking,
)
from .qed import score_qed
from .sa import score_sa
from .toxicity import score_toxicity

__all__ = [
    "get_active_oracle_name",
    "get_active_target_name",
    "score_mpro_docking",
    "score_qed",
    "score_sa",
    "score_toxicity",
]
