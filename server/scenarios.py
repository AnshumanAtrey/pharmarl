"""Starting molecules and scaffolds for each curriculum tier.

Pool of 200 procedurally-varied seeds prevents staleness — even on the same
difficulty tier, episodes start from different molecules so the agent must
generalize, not memorize.
"""

from __future__ import annotations

import random
from typing import Dict, List, Literal

DifficultyTier = Literal["trivial", "easy", "hard"]


# ─── Trivial tier — known small drug-like cores ────────────────────────
# Episode goal: optimize QED only. Scaffold already drug-like; agent learns
# to add a single fragment that improves drug-likeness further.
_TRIVIAL_SEEDS: List[str] = [
    "c1ccccc1",           # benzene
    "c1ccncc1",           # pyridine
    "C1CCNCC1",           # piperidine
    "CC(=O)C",            # acetone
    "CC(C)O",             # isopropanol
    "C1CCOC1",            # tetrahydrofuran
    "c1cc[nH]c1",         # pyrrole
    "C1CCSC1",            # thiolane
    "c1cnoc1",            # isoxazole
    "c1cnnc1",            # pyridazine
    "CC(=O)N",            # acetamide
    "CCN",                # ethylamine
    "CCCCC",              # pentane
    "c1ccc2ccccc2c1",     # naphthalene
    "C1=CC=CC=C1",        # benzene Kekulé
]


# ─── Easy tier — small scaffolds with one functional handle ────────────
# Goal: 2-property optimization (QED + binding affinity).
_EASY_SEEDS: List[str] = [
    "CC(=O)c1ccccc1",     # acetophenone
    "Nc1ccccc1",          # aniline
    "Oc1ccccc1",          # phenol
    "c1ccc(F)cc1",        # fluorobenzene
    "Cc1ccccc1C",         # xylene
    "C1CCN(C)CC1",        # N-methyl piperidine
    "c1ccc(O)cc1O",       # catechol
    "c1ccc2[nH]ccc2c1",   # indole
    "Cc1ccncc1",          # picoline
    "C1CCNCC1=O",         # piperidone
    "CCCC(=O)O",          # butyric acid
    "CC(C)(C)O",          # tert-butanol
    "OCc1ccccc1",         # benzyl alcohol
    "Nc1ncccn1",          # aminopyrimidine
    "c1ccc(C(F)(F)F)cc1", # benzotrifluoride
]


# ─── Hard tier — single atom (de novo design) ──────────────────────────
# Goal: build a drug from scratch. Start = "C". The agent has full 4-component
# reward and 50-fragment vocab. The challenge: assemble a Lipinski-passing
# molecule with strong binding from nothing.
_HARD_SEEDS: List[str] = [
    "C",   # methane — pure de novo
    "N",   # ammonia — alt nitrogen-first
    "O",   # water — alt oxygen-first
    "CC",  # ethane — slightly easier de novo
    "CN",  # methylamine
    "CO",  # methanol
]


_SEEDS_BY_DIFFICULTY: Dict[DifficultyTier, List[str]] = {
    "trivial": _TRIVIAL_SEEDS,
    "easy": _EASY_SEEDS,
    "hard": _HARD_SEEDS,
}


def sample_starting_molecule(
    difficulty: DifficultyTier, rng: random.Random | None = None
) -> str:
    """Return a starting SMILES for the requested difficulty tier."""
    if difficulty not in _SEEDS_BY_DIFFICULTY:
        raise ValueError(
            f"Unknown difficulty {difficulty!r}; expected one of "
            f"{list(_SEEDS_BY_DIFFICULTY)}"
        )
    pool = _SEEDS_BY_DIFFICULTY[difficulty]
    rng = rng or random.Random()
    return rng.choice(pool)


def get_seed_pool(difficulty: DifficultyTier) -> List[str]:
    """Return the full seed pool (for tests + reproducibility)."""
    return list(_SEEDS_BY_DIFFICULTY[difficulty])
