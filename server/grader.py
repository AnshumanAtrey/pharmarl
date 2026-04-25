"""Composite reward computation.

Per-step (dense) reward:
  +0.05  if the post-edit molecule passes Lipinski
   0.0  if it doesn't (no penalty — the next edit can recover)
  -0.5  if the action JSON was malformed (parse penalty)
  -0.1  if the action attempted but failed (e.g. invalid position)

Terminal (sparse, large) reward — issued only on TERMINATE:
  composite = 0.40 * docking
            + 0.25 * qed
            + 0.15 * sa
            + 0.20 * (1 - toxicity)
  Final reward = composite * 10  (scaled into ~0-10 range)

  Gated by Lipinski: if the final molecule fails Rule of 5, terminal reward
  is reduced by 50% — anti-reward-hacking constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .molecule_engine.validation import check_lipinski
from .oracles import score_mpro_docking, score_qed, score_sa, score_toxicity
from .rubrics import (
    BindingRubric,
    QedRubric,
    SaRubric,
    ToxicityRubric,
    composite_for_target,
)


# Reward weights (must sum to 1.0)
W_DOCKING = 0.40
W_QED = 0.25
W_SA = 0.15
W_TOX = 0.20

assert abs(W_DOCKING + W_QED + W_SA + W_TOX - 1.0) < 1e-6, "weights must sum to 1"


# Per-step shaping
LIPINSKI_BONUS = 0.05
INVALID_ACTION_PENALTY = -0.1
PARSE_FAIL_PENALTY = -0.5
TERMINAL_REWARD_SCALE = 10.0
LIPINSKI_GATE_FACTOR = 0.5  # halve terminal reward if final molecule fails Rule of 5


@dataclass
class StepReward:
    reward: float
    breakdown: Dict[str, float]


@dataclass
class TerminalReward:
    reward: float
    composite: float
    components: Dict[str, float]
    lipinski_passes: bool


def step_shaping_reward(smiles: str, action_was_valid: bool) -> StepReward:
    """Per-step reward — small dense signal that stabilizes training."""
    if not action_was_valid:
        return StepReward(reward=INVALID_ACTION_PENALTY, breakdown={"invalid_action": INVALID_ACTION_PENALTY})

    res = check_lipinski(smiles)
    if res is None or not res.passes:
        return StepReward(reward=0.0, breakdown={"lipinski": 0.0})

    return StepReward(reward=LIPINSKI_BONUS, breakdown={"lipinski": LIPINSKI_BONUS})


def parse_failure_reward() -> StepReward:
    """Reward when the agent's action JSON couldn't be parsed."""
    return StepReward(reward=PARSE_FAIL_PENALTY, breakdown={"parse_fail": PARSE_FAIL_PENALTY})


def terminal_reward(
    smiles: str,
    components_active: tuple[str, ...] = ("qed", "docking", "sa", "toxicity"),
    target: str | None = None,
    weights: tuple | None = None,
) -> TerminalReward:
    """Composite oracle reward issued on TERMINATE.

    `components_active` lets the curriculum select which oracles count
    (Trivial uses just QED; Hard uses all 4).

    `target`, when provided, routes the binding component to a specific
    classifier (DRD2 / GSK3B / JNK3) — enables multi-target training and
    held-out evaluation. None falls back to the env's default oracle.

    `weights`, when provided as ``(w_docking, w_qed, w_sa, w_tox)``, overrides
    the static module-level constants. Used by the schema-drift mechanic
    (Patronus AI sub-theme) to change reward weights mid-episode. When
    ``weights=None`` behavior is identical to before — fully backwards
    compatible.
    """
    qed = score_qed(smiles)
    docking = score_mpro_docking(smiles, target=target)
    sa = score_sa(smiles)
    tox = score_toxicity(smiles)

    components: Dict[str, float] = {
        "qed": qed,
        "docking": docking,
        "sa": sa,
        "toxicity_raw": tox,
        "toxicity_clean": 1.0 - tox,
    }

    # Resolve weights — dynamic if provided, otherwise the module defaults.
    if weights is not None:
        w_docking, w_qed, w_sa, w_tox = weights
    else:
        w_docking, w_qed, w_sa, w_tox = W_DOCKING, W_QED, W_SA, W_TOX

    # Trivial/Easy curriculum tiers preserve their legacy formulas exactly so
    # the early-training reward curve doesn't shift; Hard (and any other
    # full-component set) uses the composable rubric path so we hit the
    # OpenEnv "composable rubrics" judging criterion.
    if components_active == ("qed",):
        composite = qed
    elif set(components_active) == {"qed", "docking"}:
        denom = w_qed + w_docking
        if denom <= 0:
            composite = 0.0
        else:
            composite = (w_qed * qed + w_docking * docking) / denom
    else:
        if weights is None:
            composite_rubric = composite_for_target(target)
        else:
            composite_rubric = (
                BindingRubric(target) * w_docking
                + QedRubric() * w_qed
                + SaRubric() * w_sa
                + ToxicityRubric() * w_tox
            )
        composite = composite_rubric.score(smiles)

    # Lipinski gate
    lipinski = check_lipinski(smiles)
    lipinski_passes = lipinski is not None and lipinski.passes
    if not lipinski_passes:
        composite *= LIPINSKI_GATE_FACTOR

    final_reward = composite * TERMINAL_REWARD_SCALE
    return TerminalReward(
        reward=float(final_reward),
        composite=float(composite),
        components=components,
        lipinski_passes=bool(lipinski_passes),
    )
