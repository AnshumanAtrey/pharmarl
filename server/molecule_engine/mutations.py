"""Apply ADD_FRAGMENT / REMOVE_FRAGMENT / SUBSTITUTE_ATOM edits.

The agent's actions translate into one of these three operations on the
current molecule. SELFIES guarantees the result is chemically valid; if a
mutation cannot be applied (out-of-range position, atom mismatch, etc.) we
raise MutationError and the env scores it as last_action_valid=False with
no reward change.
"""

from __future__ import annotations

from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

from .validation import canonicalize_smiles, is_valid_molecule


class MutationError(Exception):
    """Raised when an edit cannot be applied to the current molecule."""


def _atom_count(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return mol.GetNumAtoms()


def apply_add_fragment(
    current_smiles: str,
    fragment_smiles: str,
    position: Optional[int] = None,
) -> str:
    """Attach a fragment to the current molecule at the given heavy-atom index.

    If position is None, attaches to the last atom (simplest path for the
    LLM — minimizes the number of integers it needs to reason about).
    Returns canonical SMILES of the new molecule.
    """
    if not is_valid_molecule(current_smiles):
        raise MutationError(f"current molecule is invalid: {current_smiles!r}")
    if not is_valid_molecule(fragment_smiles):
        raise MutationError(f"fragment is invalid: {fragment_smiles!r}")

    parent = Chem.RWMol(Chem.MolFromSmiles(current_smiles))
    fragment = Chem.MolFromSmiles(fragment_smiles)

    n_parent = parent.GetNumAtoms()
    if position is None:
        position = n_parent - 1
    if position < 0 or position >= n_parent:
        raise MutationError(
            f"position {position} out of range for {n_parent}-atom parent"
        )

    # Combine into one editable mol
    combined = Chem.RWMol(Chem.CombineMols(parent, fragment))
    fragment_attach_idx = n_parent  # first atom of the appended fragment

    # Add a single bond between parent[position] and fragment[0]
    try:
        combined.AddBond(position, fragment_attach_idx, Chem.BondType.SINGLE)
    except Exception as e:
        raise MutationError(f"cannot bond at position {position}: {e}") from e

    new_mol = combined.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except Exception as e:
        raise MutationError(f"resulting molecule failed sanitization: {e}") from e

    canonical = canonicalize_smiles(Chem.MolToSmiles(new_mol))
    if canonical is None:
        raise MutationError("post-merge molecule could not be canonicalized")
    return canonical


def apply_remove_fragment(current_smiles: str, position: int) -> str:
    """Remove the heavy atom at `position`. The molecule must remain connected.

    Returns canonical SMILES of the result.
    """
    if not is_valid_molecule(current_smiles):
        raise MutationError(f"current molecule is invalid: {current_smiles!r}")

    mol = Chem.RWMol(Chem.MolFromSmiles(current_smiles))
    n = mol.GetNumAtoms()
    if n <= 1:
        raise MutationError("cannot remove from a 1-atom molecule")
    if position < 0 or position >= n:
        raise MutationError(f"position {position} out of range for {n}-atom molecule")

    try:
        mol.RemoveAtom(position)
        new = mol.GetMol()
        Chem.SanitizeMol(new)
    except Exception as e:
        raise MutationError(f"removal failed: {e}") from e

    smi = Chem.MolToSmiles(new)
    # Reject disconnected fragments (REMOVE shouldn't bisect the molecule)
    if "." in smi:
        raise MutationError("removal produced disconnected fragments")

    canonical = canonicalize_smiles(smi)
    if canonical is None:
        raise MutationError("post-removal molecule could not be canonicalized")
    return canonical


def apply_substitute_atom(
    current_smiles: str,
    position: int,
    new_atom_symbol: str,
) -> str:
    """Replace the atom at `position` with `new_atom_symbol`.

    Allowed substitutions (Stage 1): C, N, O, S, F, Cl, Br.
    Returns canonical SMILES.
    """
    allowed = {"C", "N", "O", "S", "F", "Cl", "Br", "I", "P"}
    if new_atom_symbol not in allowed:
        raise MutationError(
            f"atom {new_atom_symbol!r} not in allowed set {sorted(allowed)}"
        )

    if not is_valid_molecule(current_smiles):
        raise MutationError(f"current molecule is invalid: {current_smiles!r}")

    mol = Chem.RWMol(Chem.MolFromSmiles(current_smiles))
    n = mol.GetNumAtoms()
    if position < 0 or position >= n:
        raise MutationError(f"position {position} out of range for {n}-atom molecule")

    try:
        mol.GetAtomWithIdx(position).SetAtomicNum(
            Chem.GetPeriodicTable().GetAtomicNumber(new_atom_symbol)
        )
        new = mol.GetMol()
        Chem.SanitizeMol(new)
    except Exception as e:
        raise MutationError(f"substitution failed: {e}") from e

    canonical = canonicalize_smiles(Chem.MolToSmiles(new))
    if canonical is None:
        raise MutationError("post-substitution molecule could not be canonicalized")
    return canonical
