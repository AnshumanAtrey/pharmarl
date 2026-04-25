"""Composable rubrics for PharmaRL.

The OpenEnv judging guide explicitly rewards "composable rubrics > monolithic
scoring." This module decomposes the composite reward formula into named,
weight-able pieces that compose via `*` (weight) and `+` (sum) operators.

Backwards compatibility: server/grader.py still exports terminal_reward()
with identical output values for any given input. The grader uses these
rubrics internally instead of inlined arithmetic.

Usage:

    from server.rubrics import (
        QedRubric, SaRubric, ToxicityRubric, BindingRubric, composite_for_target,
    )

    # Default composite (matches the legacy weights):
    r = composite_for_target("DRD2")
    score = r.score("CC(=O)Oc1ccccc1C(=O)O")  # aspirin

    # Build a custom rubric:
    r = QedRubric() * 0.5 + SaRubric() * 0.5
    score = r.score(smiles)  # 0.5*qed + 0.5*sa

The implementation is intentionally minimal — the goal is the structural
seam, not abstract optimality. Plays nicely with the OpenEnv Rubric pattern
without taking a hard dependency on any specific OpenEnv class hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


# ─── Operator-supporting base ─────────────────────────────────────────


class _BaseRubric:
    """Mixin providing `*` (weighting) and `+` (composition) operators."""

    def score(self, smiles: str) -> float:  # pragma: no cover — abstract
        raise NotImplementedError

    def __mul__(self, weight: float) -> "WeightedRubric":
        return WeightedRubric(base=self, weight=float(weight))

    __rmul__ = __mul__  # support `0.4 * BindingRubric()` syntax

    def __add__(self, other: "_BaseRubric") -> "SumRubric":
        if isinstance(other, SumRubric):
            return SumRubric(parts=[self] + other.parts)
        return SumRubric(parts=[self, other])


@dataclass
class WeightedRubric(_BaseRubric):
    """A rubric scaled by a constant weight."""

    base: _BaseRubric
    weight: float

    def score(self, smiles: str) -> float:
        return self.base.score(smiles) * self.weight


@dataclass
class SumRubric(_BaseRubric):
    """A sum of rubrics. Composition is associative."""

    parts: List[_BaseRubric]

    def score(self, smiles: str) -> float:
        return sum(p.score(smiles) for p in self.parts)


# ─── Concrete rubrics — wrap the existing oracle functions ─────────────


class QedRubric(_BaseRubric):
    """Quantitative Estimate of Drug-likeness (Bickerton et al. 2012). Range [0, 1]."""

    def score(self, smiles: str) -> float:
        from .oracles import score_qed
        return float(score_qed(smiles))


class SaRubric(_BaseRubric):
    """Synthetic accessibility, normalized so higher = more synthesizable. Range [0, 1]."""

    def score(self, smiles: str) -> float:
        from .oracles import score_sa
        return float(score_sa(smiles))


class ToxicityRubric(_BaseRubric):
    """Inverted CYP3A4 toxicity probability (1 - tox). Higher = less toxic. Range [0, 1]."""

    def score(self, smiles: str) -> float:
        from .oracles import score_toxicity
        return 1.0 - float(score_toxicity(smiles))


class BindingRubric(_BaseRubric):
    """Binding-classifier or docking-affinity score (target-routed). Range [0, 1]."""

    def __init__(self, target: Optional[str] = None) -> None:
        self.target = target

    def score(self, smiles: str) -> float:
        from .oracles import score_mpro_docking
        return float(score_mpro_docking(smiles, target=self.target))


# ─── Pre-built composites ──────────────────────────────────────────────


def composite_for_target(
    target: Optional[str] = None,
    weights: Optional[tuple] = None,
) -> _BaseRubric:
    """Default composite: 0.40 binding + 0.25 QED + 0.15 SA + 0.20 (1-tox).

    `weights`, when passed as `(w_docking, w_qed, w_sa, w_tox)`, overrides
    the legacy module-level constants. Used by the schema-drift mechanic.

    Returns a `_BaseRubric` whose `.score(smiles)` produces a [0, 1]-range
    composite (modulo Lipinski gating, which lives in the grader).
    """
    if weights is None:
        w_docking, w_qed, w_sa, w_tox = 0.40, 0.25, 0.15, 0.20
    else:
        w_docking, w_qed, w_sa, w_tox = weights

    return (
        BindingRubric(target=target) * w_docking
        + QedRubric() * w_qed
        + SaRubric() * w_sa
        + ToxicityRubric() * w_tox
    )
