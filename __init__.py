"""PharmaRL — first OpenEnv-native LLM-as-policy environment for iterative molecular optimization.

Stage 1: DRD2 (canonical MOSES/GuacaMol molecular RL benchmark) — pure-Python deps.
Stage 2: pyscreener-backed docking (NSP15/EGFR/ABL/BACE1) when PHARMARL_ENABLE_DOCKING=1
         and OpenBabel + AutoDock Vina are available on the host.
"""

from .client import PharmaRLEnv
from .models import (
    DifficultyTier,
    MoleculeAction,
    MoleculeActionType,
    MoleculeObservation,
    MoleculeState,
)

__all__ = [
    "PharmaRLEnv",
    "MoleculeAction",
    "MoleculeActionType",
    "MoleculeObservation",
    "MoleculeState",
    "DifficultyTier",
]
