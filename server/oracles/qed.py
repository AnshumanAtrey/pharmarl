"""QED — Quantitative Estimate of Drug-likeness (Bickerton et al. 2012).

A weighted composite of 8 molecular descriptors (MW, LogP, HBD, HBA,
PSA, rotatable bonds, aromatic rings, structural alerts). Returns 0-1.

Reference values:
    aspirin  ≈ 0.55
    caffeine ≈ 0.55
    ibuprofen ≈ 0.81
    morphine  ≈ 0.65
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import QED


def score_qed(smiles: str) -> float:
    """Returns QED score in [0, 1]. Returns 0.0 for invalid molecules.

    Note: RDKit's MolFromSmiles('') returns a 0-atom Mol (not None) and QED.qed()
    on it gives ~0.34 — a reward-hacking surface for empty/invalid outputs.
    Explicit zero-atom check below.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return 0.0
    try:
        return float(QED.qed(mol))
    except Exception:
        return 0.0
