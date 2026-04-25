"""Curriculum controller — RLVE adaptive difficulty.

Maps GRPO training step → difficulty tier. Trivial guarantees a moving
reward curve early; the env auto-promotes once the trainer demonstrates
mastery. Hits Theme 4 (Self-Improvement) of the judging rubric.

Trainer can call `pick_difficulty(step)` to get the tier for the next
episode, or `pick_difficulty()` for a uniform random sample (eval mode).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Optional

DifficultyTier = Literal["trivial", "easy", "hard"]


@dataclass(frozen=True)
class CurriculumConfig:
    trivial_until: int = 100   # GRPO step boundary
    easy_until: int = 300
    # >= easy_until -> hard

    # Action vocab + max_steps per tier (canonical mapping)
    trivial_max_steps: int = 10
    easy_max_steps: int = 15
    hard_max_steps: int = 20

    # Reward components active at each tier
    trivial_components: tuple = ("qed",)
    easy_components: tuple = ("qed", "docking")
    hard_components: tuple = ("qed", "docking", "sa", "toxicity")

    # Multi-target setup for the held-out generalization test.
    # We train on `training_targets` (rotating per episode) and reserve
    # `held_out_target` for the final eval — measures whether learned
    # medicinal-chemistry primitives transfer to a target the model never saw.
    training_targets: tuple = ("DRD2", "GSK3B")
    held_out_target: str = "JNK3"

    # ─── Schema drift (Patronus AI sub-theme) ──────────────────────────────
    # Mid-episode reward weight changes, modeling the real medicinal-chemistry
    # workflow where a project starts optimizing one objective and the
    # constraints shift mid-development (e.g. potency push uncovers an ADMET
    # liability, synthesizability tightens before scale-up).
    #
    # Default OFF — flagging this on is opt-in via reset(schema_drift=True)
    # or by constructing the env with a config that has schema_drift_enabled=True.
    schema_drift_enabled: bool = False                  # MASTER FLAG, default OFF
    drift_profiles: tuple = (                            # sampled per-episode when enabled
        "static",                                        # no drift (control)
        "early_admet",                                   # ADMET kicks in mid-episode
        "late_potency",                                  # potency requirement added mid-episode
    )
    drift_step: int = 8                                  # step at which the drift fires
    # (w_docking, w_qed, w_sa, w_tox)
    weights_static: tuple = (0.40, 0.25, 0.15, 0.20)               # current default
    weights_early_admet_pre: tuple = (0.60, 0.40, 0.00, 0.00)      # binding+drug-like only
    weights_early_admet_post: tuple = (0.30, 0.20, 0.25, 0.25)     # SA + tox activate
    weights_late_potency_pre: tuple = (0.10, 0.40, 0.30, 0.20)     # drug-like + ADMET focus
    weights_late_potency_post: tuple = (0.45, 0.20, 0.20, 0.15)    # potency now matters

    # ─── Multi-actor critic (Halluminate sub-theme) ────────────────────────
    # Rules-based medicinal-chemist critic inspects each post-edit molecule
    # and the critique is appended to the next observation's metadata. The
    # agent can integrate or ignore the feedback. Default OFF so the headline
    # training run is unaffected.
    critic_enabled: bool = False                        # MASTER FLAG, default OFF


DEFAULT_CONFIG = CurriculumConfig()


def pick_training_target(
    rng: random.Random | None = None,
    config: CurriculumConfig = DEFAULT_CONFIG,
) -> str:
    """Pick a training target (rotates through training_targets uniformly)."""
    rng = rng or random.Random()
    return rng.choice(config.training_targets)


def pick_difficulty(
    training_step: Optional[int] = None,
    config: CurriculumConfig = DEFAULT_CONFIG,
    rng: random.Random | None = None,
) -> DifficultyTier:
    """Return the curriculum tier for the next episode.

    If training_step is None → uniform random across tiers (used at eval/demo).
    """
    if training_step is None:
        rng = rng or random.Random()
        return rng.choice(("trivial", "easy", "hard"))

    if training_step < config.trivial_until:
        return "trivial"
    if training_step < config.easy_until:
        return "easy"
    return "hard"


def max_steps_for(difficulty: DifficultyTier, config: CurriculumConfig = DEFAULT_CONFIG) -> int:
    return {
        "trivial": config.trivial_max_steps,
        "easy": config.easy_max_steps,
        "hard": config.hard_max_steps,
    }[difficulty]


def reward_components_for(
    difficulty: DifficultyTier, config: CurriculumConfig = DEFAULT_CONFIG
) -> tuple[str, ...]:
    return {
        "trivial": config.trivial_components,
        "easy": config.easy_components,
        "hard": config.hard_components,
    }[difficulty]


# ─── Schema drift helpers (Patronus AI sub-theme) ──────────────────────────


def pick_drift_profile(
    rng: random.Random | None = None,
    config: CurriculumConfig = DEFAULT_CONFIG,
) -> str:
    """Pick the drift profile for the next episode.

    Returns ``"static"`` (no drift) when schema drift is disabled — guarantees
    the headline training run is unaffected unless the master flag is on.
    """
    if not config.schema_drift_enabled:
        return "static"
    rng = rng or random.Random()
    return rng.choice(config.drift_profiles)


def weights_for(
    profile: str,
    step_count: int,
    drift_step: int,
    config: CurriculumConfig = DEFAULT_CONFIG,
) -> tuple:
    """Return ``(w_docking, w_qed, w_sa, w_tox)`` for the current step.

    - ``static`` always returns ``config.weights_static``.
    - ``early_admet`` / ``late_potency`` return their ``pre`` weights for
      ``step_count < drift_step`` and ``post`` weights once the drift fires.

    Unknown profiles fall back to the static weights — defensive default that
    keeps the headline run safe if a typo slips into a config.
    """
    if profile == "early_admet":
        if step_count < drift_step:
            return config.weights_early_admet_pre
        return config.weights_early_admet_post
    if profile == "late_potency":
        if step_count < drift_step:
            return config.weights_late_potency_pre
        return config.weights_late_potency_post
    # "static" or unknown -> static
    return config.weights_static
