"""Curriculum controller — RLVE adaptive difficulty.

Two modes:
  1. `pick_difficulty(step)` — step-threshold schedule (deterministic, simple).
  2. `AdaptiveCurriculum` — promotes/demotes based on rolling success rate
     over the last N rollouts. This is the RLVE pattern from the hackathon
     guide (Q22, Q35): keep the agent near its capability frontier instead
     of stalling on tasks that are too easy or too hard.

The trainer chooses which to use. Both return a `DifficultyTier`.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Literal, Optional

DifficultyTier = Literal["trivial", "easy", "hard"]
_TIERS: tuple[DifficultyTier, ...] = ("trivial", "easy", "hard")


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

    # Adaptive curriculum knobs (ignored by `pick_difficulty`)
    adaptive_window: int = 16          # rolling window of cumulative rewards
    promote_threshold: float = 0.6     # rolling mean ≥ this → promote tier
    demote_threshold: float = -0.3     # rolling mean < this → drop tier
    min_steps_per_tier: int = 24       # don't oscillate; stay in tier ≥ this many rollouts


DEFAULT_CONFIG = CurriculumConfig()


def pick_difficulty(
    training_step: Optional[int] = None,
    config: CurriculumConfig = DEFAULT_CONFIG,
    rng: random.Random | None = None,
) -> DifficultyTier:
    """Step-threshold schedule. None → uniform random (eval/demo mode)."""
    if training_step is None:
        rng = rng or random.Random()
        return rng.choice(_TIERS)

    if training_step < config.trivial_until:
        return "trivial"
    if training_step < config.easy_until:
        return "easy"
    return "hard"


@dataclass
class AdaptiveCurriculum:
    """Success-rate-driven tier promotion (RLVE-style).

    Usage:
        curr = AdaptiveCurriculum()
        for step in range(N):
            difficulty = curr.next_tier()
            ... rollout ...
            curr.record(rollout_cumulative_reward)

    The tier only moves when (a) the rolling window is full and (b) we've
    spent at least `min_steps_per_tier` rollouts in the current tier — this
    prevents oscillation. Demotion triggers if rewards collapse (catastrophic
    forgetting after promotion) and gives the agent a chance to recover.
    """

    config: CurriculumConfig = DEFAULT_CONFIG
    _tier_idx: int = 0
    _rollouts_in_tier: int = 0
    _window: Deque[float] = field(default_factory=lambda: deque(maxlen=16))

    def __post_init__(self) -> None:
        # Honor the configured window size (deque was created with default 16)
        self._window = deque(maxlen=self.config.adaptive_window)

    @property
    def current_tier(self) -> DifficultyTier:
        return _TIERS[self._tier_idx]

    @property
    def rolling_mean(self) -> float:
        return sum(self._window) / len(self._window) if self._window else 0.0

    def next_tier(self) -> DifficultyTier:
        """Return the tier for the upcoming rollout (does not record)."""
        return self.current_tier

    def record(self, cumulative_reward: float) -> dict:
        """Record one rollout's cumulative reward; maybe promote/demote.

        Returns a small dict with diagnostic info (current tier, rolling
        mean, whether we just transitioned) — useful for W&B.
        """
        self._window.append(float(cumulative_reward))
        self._rollouts_in_tier += 1
        transitioned = False
        old_tier = self.current_tier

        ready = (
            len(self._window) == self._window.maxlen
            and self._rollouts_in_tier >= self.config.min_steps_per_tier
        )
        if ready:
            mean = self.rolling_mean
            if mean >= self.config.promote_threshold and self._tier_idx < len(_TIERS) - 1:
                self._tier_idx += 1
                transitioned = True
            elif mean < self.config.demote_threshold and self._tier_idx > 0:
                self._tier_idx -= 1
                transitioned = True
            if transitioned:
                self._rollouts_in_tier = 0
                self._window.clear()

        return {
            "tier": self.current_tier,
            "rolling_mean": self.rolling_mean,
            "rollouts_in_tier": self._rollouts_in_tier,
            "transitioned_from": old_tier if transitioned else None,
        }


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
