"""Molecule engine — SELFIES editing primitives.

Three pieces:
  - fragments: vocabulary of building-block SMILES
  - mutations: ADD/REMOVE/SUBSTITUTE applied to a current molecule
  - validation: chemical validity + Lipinski Rule of 5
"""

from .fragments import FRAGMENT_VOCAB, get_vocab_for_difficulty
from .mutations import (
    MutationError,
    apply_add_fragment,
    apply_remove_fragment,
    apply_substitute_atom,
)
from .validation import (
    LipinskiResult,
    canonicalize_smiles,
    check_lipinski,
    is_valid_molecule,
    smiles_to_selfies,
    selfies_to_smiles,
)

__all__ = [
    "FRAGMENT_VOCAB",
    "get_vocab_for_difficulty",
    "MutationError",
    "apply_add_fragment",
    "apply_remove_fragment",
    "apply_substitute_atom",
    "LipinskiResult",
    "canonicalize_smiles",
    "check_lipinski",
    "is_valid_molecule",
    "smiles_to_selfies",
    "selfies_to_smiles",
]
