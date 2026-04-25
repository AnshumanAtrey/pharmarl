"""Fragment vocabulary for molecular edits.

Three difficulty tiers with progressively larger vocabularies. Each fragment
is a valid SMILES substructure that can be attached to the current molecule.

Vocabulary curated from BRICS-fragmented FDA-approved drugs (well-known
medicinal chemistry building blocks).
"""

from __future__ import annotations

from typing import Dict, List, Literal

DifficultyTier = Literal["trivial", "easy", "hard"]


# ─── Tier 1: Trivial (5 fragments) ──────────────────────────────────────
# Goal: guarantee a moving QED reward curve in <100 GRPO steps.
# All fragments boost drug-likeness when attached to a small scaffold.
TRIVIAL_FRAGMENTS: List[str] = [
    "C",          # methyl — generic alkyl
    "O",          # hydroxyl source
    "N",          # amine source
    "c1ccccc1",   # benzene — most common drug ring
    "C(=O)O",     # carboxyl — drug-like polar group
]


# ─── Tier 2: Easy (15 fragments) ────────────────────────────────────────
# Adds heterocycles + functional groups common in protease inhibitors.
EASY_FRAGMENTS: List[str] = TRIVIAL_FRAGMENTS + [
    "c1ccncc1",      # pyridine — kinase/protease scaffold
    "C1CCNCC1",      # piperidine — found in 50%+ of approved drugs
    "C1CCOCC1",      # tetrahydropyran
    "F",             # fluorine — common bioisostere
    "Cl",            # chlorine
    "C(=O)N",        # amide — peptide bond surrogate
    "S(=O)(=O)N",    # sulfonamide
    "C#N",           # nitrile
    "C(F)(F)F",      # trifluoromethyl
    "OC",            # methoxy
]


# ─── Tier 3: Hard (50 fragments) ────────────────────────────────────────
# Full medicinal-chemistry vocabulary — Mpro inhibitor pharmacophores
# included (peptidomimetic warheads, P1/P2 substitutions, etc.).
HARD_FRAGMENTS: List[str] = EASY_FRAGMENTS + [
    # Aromatics
    "c1ccc2ccccc2c1",     # naphthalene
    "c1cc2ccccc2cn1",     # quinoline
    "c1ccc2[nH]ccc2c1",   # indole
    "c1cnc[nH]1",         # imidazole
    "c1ccoc1",            # furan
    "c1ccsc1",            # thiophene
    "c1cnc2[nH]cnc2c1",   # purine

    # Saturated heterocycles
    "C1CCNC1",            # pyrrolidine
    "C1COCCN1",           # morpholine
    "C1CCNCCN1",          # piperazine
    "N1CCCC1=O",          # 2-pyrrolidinone
    "C1CCSC1",            # tetrahydrothiophene

    # Mpro-relevant warheads / pharmacophores
    "C(=O)C=O",           # alpha-ketoaldehyde (covalent warhead)
    "C(=O)C(F)(F)F",      # trifluoroacetyl
    "C(=O)N1CCCC1",       # pyrrolidine-amide (P1 surrogate)
    "C(C)(C)C",           # tert-butyl (lipophilic block)
    "C1CC1",              # cyclopropyl

    # Polar / hydrogen-bond groups
    "OS(=O)(=O)O",        # sulfonate
    "NS(=O)(=O)C",        # sulfonamide N
    "C(=O)NO",            # hydroxamic acid
    "P(=O)(O)O",          # phosphonate
    "C(=N)N",             # guanidine

    # Halogens / bioisosteres
    "Br",
    "I",
    "[O-]",
    "[N+]",

    # Common substituents
    "CC",                 # ethyl
    "CCC",                # propyl
    "CCCC",               # butyl
    "CO",                 # methanol
    "CCO",                # ethanol-attached
    "C=C",                # vinyl
    "C#C",                # alkynyl

    # Complex pharmacophores
    "c1nc2ccccc2[nH]1",   # benzimidazole
    "c1ccc2ncccc2c1",     # quinoline isomer
    "C1CCCCC1",           # cyclohexane
]


FRAGMENT_VOCAB: Dict[DifficultyTier, List[str]] = {
    "trivial": TRIVIAL_FRAGMENTS,
    "easy": EASY_FRAGMENTS,
    "hard": HARD_FRAGMENTS,
}


def get_vocab_for_difficulty(difficulty: DifficultyTier) -> List[str]:
    """Return the fragment vocabulary for a given difficulty tier."""
    if difficulty not in FRAGMENT_VOCAB:
        raise ValueError(
            f"Unknown difficulty {difficulty!r}. "
            f"Expected one of: {list(FRAGMENT_VOCAB)}"
        )
    return FRAGMENT_VOCAB[difficulty]
