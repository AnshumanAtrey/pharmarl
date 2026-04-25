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
        default="SARS-CoV-2_Mpro",
        description="Current docking target this episode is optimizing against.",
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


class MoleculeState(State):
    """Full episode state. Persists across step calls within one episode."""

    target: str = Field(default="SARS-CoV-2_Mpro")
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
