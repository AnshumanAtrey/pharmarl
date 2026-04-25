"""Rules-based medicinal chemist critic.

After the agent edits a molecule, the critic examines the result and emits
a structured critique. Hits the Halluminate "multi-actor" sub-theme without
the cost/latency of a real LLM call.

The critic is a DETERMINISTIC rules engine — same input always gives the
same critique. That's intentional: it makes critic-conditioned training
reproducible and the critic itself non-gameable in a stochastic sense.

Design notes:
  - Each rule is a SMARTS substructure search OR an RDKit Descriptors check.
  - Issues are tagged with a stable code (e.g. PAINS_THIOCARBONYL_NUISANCE) so
    the policy can learn to recognize and respond to specific critique types.
  - Verdict: 'reject' on hard error, 'revise' on >=2 warnings, else 'approve'.
  - Latency: ms-scale, no I/O, fully in-process. Future work could swap in a
    frozen LLM critic at this seam without changing the env contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors


@dataclass
class CritiqueIssue:
    severity: str   # "warning" | "error" | "info"
    code: str       # e.g. "PAINS_THIOCARBONYL_NUISANCE", "MW_TOO_HIGH"
    message: str    # human-readable explanation


@dataclass
class Critique:
    smiles: str
    issues: List[CritiqueIssue] = field(default_factory=list)
    overall: str = "approve"   # "approve" | "revise" | "reject"


# Substructure SMARTS for known PAINS / nuisance moieties.
# Extend this catalog as the curriculum encounters new failure modes.
_KNOWN_PAINS_SMARTS: Dict[str, str] = {
    "thiocarbonyl_nuisance": "C(=S)",                       # known assay interference
    "rhodanine_core": "C1SC(=N1)NC=O",                      # PAINS rhodanine
    "michael_acceptor_nitroaromatic": "[N+](=O)[O-]",       # nitro group, often reactive
}


# Reactive / metabolically labile groups — flagged as "info", not "warning",
# because some are tolerable in a hit-finding context but worth surfacing.
_REACTIVE_GROUPS: Dict[str, str] = {
    "alkyl_halide": "[CX4][F,Cl,Br,I]",
    "epoxide": "C1CO1",
    "anhydride": "C(=O)OC(=O)",
}


class MedChemCritic:
    """Rules-based reviewer. Inspects a molecule and returns structured critique.

    A separate logical agent that runs after the policy proposes an edit. The
    env appends the critique to the next observation; the policy can choose to
    revise (REMOVE_FRAGMENT / SUBSTITUTE_ATOM) or proceed.
    """

    def critique(self, smiles: str) -> Critique:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None or mol.GetNumAtoms() == 0:
            return Critique(
                smiles=smiles,
                issues=[CritiqueIssue("error", "INVALID", "Could not parse molecule")],
                overall="reject",
            )

        issues: List[CritiqueIssue] = []

        # ── Lipinski-flavored property checks ──────────────────────────────
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        if mw > 500:
            issues.append(CritiqueIssue(
                "warning", "MW_TOO_HIGH",
                f"MW={mw:.1f} > 500 — Lipinski violation, oral bioavailability concern",
            ))
        if logp > 5:
            issues.append(CritiqueIssue(
                "warning", "LOGP_TOO_HIGH",
                f"LogP={logp:.2f} > 5 — likely poor solubility",
            ))

        # ── PAINS substructure flags ───────────────────────────────────────
        for code, smarts in _KNOWN_PAINS_SMARTS.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None and mol.HasSubstructMatch(patt):
                issues.append(CritiqueIssue(
                    "warning", f"PAINS_{code.upper()}",
                    f"Contains pattern {code} — known assay interference / nuisance",
                ))

        # ── Reactive group flags (info, not warning) ───────────────────────
        for code, smarts in _REACTIVE_GROUPS.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt is not None and mol.HasSubstructMatch(patt):
                issues.append(CritiqueIssue(
                    "info", f"REACTIVE_{code.upper()}",
                    f"Contains {code} — be aware of reactivity / metabolic liability",
                ))

        # ── Heavy-atom sanity ──────────────────────────────────────────────
        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy < 5:
            issues.append(CritiqueIssue(
                "info", "TOO_SMALL",
                f"Only {n_heavy} heavy atoms — likely too small for selective binding",
            ))

        # ── Verdict ────────────────────────────────────────────────────────
        n_errors = sum(1 for i in issues if i.severity == "error")
        n_warnings = sum(1 for i in issues if i.severity == "warning")
        if n_errors > 0:
            overall = "reject"
        elif n_warnings >= 2:
            overall = "revise"
        else:
            overall = "approve"

        return Critique(smiles=smiles, issues=issues, overall=overall)


# Singleton — env imports this instance to avoid re-instantiation per step.
default_critic = MedChemCritic()


def critique_to_dict(c: Critique) -> Dict[str, object]:
    """Serialize a Critique into a dict suitable for observation.metadata."""
    return {
        "overall": c.overall,
        "issues": [
            {"severity": i.severity, "code": i.code, "message": i.message}
            for i in c.issues
        ],
        "summary": f"Critic says: {c.overall}. {len(c.issues)} issues flagged.",
    }


__all__ = [
    "CritiqueIssue",
    "Critique",
    "MedChemCritic",
    "default_critic",
    "critique_to_dict",
]
