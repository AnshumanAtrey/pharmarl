"""Pydantic data models for PharmaRL.

The agent's interface to the env — every action it issues, every observation
it sees, and the full episode state are defined here.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

MoleculeActionType = Literal[
    "ADD_FRAGMENT",
    "REMOVE_FRAGMENT",
    "SUBSTITUTE_ATOM",
    "TERMINATE",
]

DifficultyTier = Literal["trivial", "easy", "hard"]


class MoleculeAction(Action):
    """One molecular edit issued by the agent.

    The model emits structured JSON parsed into this shape. Malformed JSON
    yields a -0.5 parse penalty (via the grader) instead of a hard failure.
    """

    action_type: MoleculeActionType = Field(
        ...,
        description="Edit type. TERMINATE submits the current molecule for terminal reward.",
    )

    fragment: Optional[str] = Field(
        default=None,
        description="Fragment SMILES to add (required for ADD_FRAGMENT). Must be in vocab.",
    )

    position: Optional[int] = Field(
        default=None,
        description="0-indexed atom position to apply the edit at.",
    )

    new_atom: Optional[str] = Field(
        default=None,
        description="Replacement atom symbol (required for SUBSTITUTE_ATOM). E.g. 'F', 'N'.",
    )


class MoleculeObservation(Observation):
    """What the agent sees after each step."""

    smiles: str = Field(
        default="",
        description="Current molecule as canonical SMILES (human-readable).",
    )

    selfies: str = Field(
        default="",
        description="Current molecule as SELFIES (always chemically valid).",
    )

    target: str = Field(
        default="DRD2_dopamine_D2_receptor",
        description=(
            "Biological target the active oracle is scoring against. Defaults to "
            "DRD2 (the canonical MOSES/GuacaMol classifier benchmark). With "
            "PHARMARL_ENABLE_DOCKING=1 and pyscreener+Vina installed, "
            "switches to a docking target (NSP15/EGFR/ABL/BACE1)."
        ),
    )

    difficulty: DifficultyTier = Field(
        default="trivial",
        description="Curriculum tier for this episode.",
    )

    properties: Dict[str, float] = Field(
        default_factory=dict,
        description="Current oracle scores: {qed, sa, docking, toxicity, mw, logp}.",
    )

    valid_actions: List[str] = Field(
        default_factory=list,
        description="Allowed action_types in current state (e.g. TERMINATE only after step 1).",
    )

    available_fragments: List[str] = Field(
        default_factory=list,
        description="Fragment vocab for current difficulty (5 / 15 / 50 SMILES).",
    )

    steps_remaining: int = Field(
        default=0,
        description="Steps left before truncation.",
    )

    last_action_valid: bool = Field(
        default=True,
        description="False if the previous edit was rejected (invalid position, broken chemistry, etc).",
    )

    message: str = Field(
        default="",
        description="Human-readable status — what just happened, why reward changed.",
    )

    truncated: bool = Field(
        default=False,
        description="True if episode ended via step limit, False if via TERMINATE.",
    )

    active_constraints: List[str] = Field(
        default_factory=list,
        description=(
            "Names of reward components that currently dominate (weight > 0). "
            "Schema-drift profiles change which components are active mid-episode."
        ),
    )

    drift_warning: str = Field(
        default="",
        description=(
            "Non-empty on the step at which the schema-drift weights flip. "
            "Surfaces the 'underlying rules just changed' signal to the agent."
        ),
    )

    critique: Optional[Dict] = Field(
        default=None,
        description=(
            "Halluminate sub-theme — populated when CurriculumConfig.critic_enabled. "
            "Structured medicinal-chemist critique: {overall, issues, summary}. "
            "Top-level field (not metadata) so it survives HTTP serialization."
        ),
    )

    oversight: Optional[Dict] = Field(
        default=None,
        description=(
            "Fleet AI sub-theme — populated at episode end when "
            "CurriculumConfig.oversight_enabled. Structured oversight report: "
            "{strategy_summary, risk_flags, risk_level, explanation, model_name}."
        ),
    )


class MoleculeState(State):
    """Full episode state. Persists across step calls within one episode."""

    target: str = Field(default="DRD2_dopamine_D2_receptor")
    difficulty: DifficultyTier = Field(default="trivial")
    max_steps: int = Field(default=10)

    smiles: str = Field(default="C", description="Current canonical SMILES.")
    selfies: str = Field(default="[C]", description="Current SELFIES string.")

    starting_smiles: str = Field(default="C", description="Scaffold this episode began with.")

    edit_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Sequence of {action_type, smiles_before, smiles_after} for last 5 edits.",
    )

    cumulative_reward: float = Field(default=0.0)
    final_oracle_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Populated only on TERMINATE — the composite reward breakdown.",
    )

    drift_profile: str = Field(
        default="static",
        description=(
            "Schema-drift profile for this episode (static / early_admet / "
            "late_potency). 'static' = fixed weights = legacy behavior."
        ),
    )
    drift_step: int = Field(
        default=0,
        description="Step on which the drift fires (post-weights take effect at this step).",
    )
