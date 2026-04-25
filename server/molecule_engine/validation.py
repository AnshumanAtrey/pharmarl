"""Chemical validity + Lipinski Rule of 5 checks.

Lipinski Rule of 5 (1997, Pfizer):
    MW   <= 500 Da
    LogP <= 5
    HBD  <= 5
    HBA  <= 10

Drugs that pass have higher oral bioavailability. Used as anti-reward-hacking
constraint: agent gets shaping bonus only when its molecule passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


@dataclass
class LipinskiResult:
    mw: float
    logp: float
    hbd: int
    hba: int
    passes: bool
    violations: int


def is_valid_molecule(smiles: str) -> bool:
    """Returns True if SMILES parses to a valid RDKit molecule."""
    if not smiles:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Returns canonical SMILES, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_selfies(smiles: str) -> Optional[str]:
    """Convert SMILES → SELFIES (always chemically valid form)."""
    try:
        return sf.encoder(smiles)
    except Exception:
        return None


def selfies_to_smiles(selfies: str) -> Optional[str]:
    """Convert SELFIES → SMILES."""
    try:
        return sf.decoder(selfies)
    except Exception:
        return None


def check_lipinski(smiles: str) -> Optional[LipinskiResult]:
    """Compute Lipinski Rule of 5 properties.

    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    violations = sum(
        [
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        ]
    )

    return LipinskiResult(
        mw=float(mw),
        logp=float(logp),
        hbd=int(hbd),
        hba=int(hba),
        passes=violations == 0,
        violations=violations,
    )
