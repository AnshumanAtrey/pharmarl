"""PharmaRL OpenEnv client.

Lets the trainer (Colab) talk to the env (HF Space) over HTTP.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import MoleculeAction, MoleculeObservation, MoleculeState


class PharmaRLEnv(EnvClient[MoleculeAction, MoleculeObservation, MoleculeState]):
    """Client for PharmaRL.

    Example:
        >>> with PharmaRLEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(difficulty="trivial")
        ...     print(result.observation.smiles)
        ...     result = env.step(MoleculeAction(
        ...         action_type="ADD_FRAGMENT",
        ...         fragment="c1ccccc1",
        ...     ))
        ...     print(result.reward, result.observation.message)
    """

    def _step_payload(self, action: MoleculeAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[MoleculeObservation]:
        obs_data = payload.get("observation", {})
        observation = MoleculeObservation(
            smiles=obs_data.get("smiles", ""),
            selfies=obs_data.get("selfies", ""),
            target=obs_data.get("target", "SARS-CoV-2_Mpro"),
            difficulty=obs_data.get("difficulty", "trivial"),
            properties=obs_data.get("properties", {}),
            valid_actions=obs_data.get("valid_actions", []),
            available_fragments=obs_data.get("available_fragments", []),
            steps_remaining=obs_data.get("steps_remaining", 0),
            last_action_valid=obs_data.get("last_action_valid", True),
            message=obs_data.get("message", ""),
            truncated=obs_data.get("truncated", False),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> MoleculeState:
        return MoleculeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            target=payload.get("target", "SARS-CoV-2_Mpro"),
            difficulty=payload.get("difficulty", "trivial"),
            max_steps=payload.get("max_steps", 10),
            smiles=payload.get("smiles", ""),
            selfies=payload.get("selfies", ""),
            starting_smiles=payload.get("starting_smiles", ""),
            edit_history=payload.get("edit_history", []),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            final_oracle_scores=payload.get("final_oracle_scores"),
        )
