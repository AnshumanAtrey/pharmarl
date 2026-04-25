"""Drug Discovery Environment — the core reset/step logic.

Implements the OpenEnv `Environment` contract. Owns episode state, applies
agent actions to the molecule, validates chemistry, and computes rewards.

Episode lifecycle:
  reset(difficulty?) -> Observation w/ seed molecule
  step(action)       -> Observation w/ mutated molecule + reward
    repeat until TERMINATE or max_steps
  reset()            -> next episode
"""

from __future__ import annotations

import logging
import random
from copy import deepcopy
from typing import Any, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        DifficultyTier,
        MoleculeAction,
        MoleculeObservation,
        MoleculeState,
    )
except ImportError:
    from models import (  # type: ignore
        DifficultyTier,
        MoleculeAction,
        MoleculeObservation,
        MoleculeState,
    )

try:
    from .curriculum import (
        CurriculumConfig,
        DEFAULT_CONFIG,
        max_steps_for,
        pick_difficulty,
        reward_components_for,
    )
    from .grader import (
        parse_failure_reward,
        step_shaping_reward,
        terminal_reward,
    )
    from .molecule_engine import (
        MutationError,
        apply_add_fragment,
        apply_remove_fragment,
        apply_substitute_atom,
        canonicalize_smiles,
        check_lipinski,
        get_vocab_for_difficulty,
        smiles_to_selfies,
    )
    from .oracles import (
        DEFAULT_TARGET,
        KNOWN_TARGETS,
        get_active_target_name,
        get_target_full_name,
    )
    from .scenarios import sample_starting_molecule
except ImportError:
    from server.curriculum import (  # type: ignore
        CurriculumConfig,
        DEFAULT_CONFIG,
        max_steps_for,
        pick_difficulty,
        reward_components_for,
    )
    from server.grader import (  # type: ignore
        parse_failure_reward,
        step_shaping_reward,
        terminal_reward,
    )
    from server.molecule_engine import (  # type: ignore
        MutationError,
        apply_add_fragment,
        apply_remove_fragment,
        apply_substitute_atom,
        canonicalize_smiles,
        check_lipinski,
        get_vocab_for_difficulty,
        smiles_to_selfies,
    )
    from server.oracles import (  # type: ignore
        DEFAULT_TARGET,
        KNOWN_TARGETS,
        get_active_target_name,
        get_target_full_name,
    )
    from server.scenarios import sample_starting_molecule  # type: ignore

logger = logging.getLogger(__name__)


class DrugDiscoveryEnvironment(Environment):
    """Multi-step molecular editing env (Stage 1 default: DRD2 binding).

    State is held internally per episode. step() mutates that state.

    The active target is determined dynamically by the oracles module — Stage 1
    serves DRD2 (no native chemistry deps); with PHARMARL_ENABLE_DOCKING=1 plus
    pyscreener+Vina installed, the binding oracle falls forward to a docking
    target (NSP15/EGFR/ABL/BACE1) and every observation's `target` field is
    updated accordingly.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, seed: Optional[int] = None, config: CurriculumConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self._rng = random.Random(seed)
        self._config = config
        self._state = MoleculeState()
        self._edit_history_full: List[dict] = []
        self._final_oracle_scores: Optional[dict] = None
        # Short routing name for the binding oracle (DRD2/GSK3B/JNK3). Resolved
        # at /reset; the human-readable full name lives in self._state.target.
        self._target_short: str = DEFAULT_TARGET

    # ─── reset / step ───────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> MoleculeObservation:
        """Start a new episode. Optional kwargs: difficulty, training_step, target."""
        difficulty: Optional[DifficultyTier] = kwargs.get("difficulty")
        training_step: Optional[int] = kwargs.get("training_step")
        target: Optional[str] = kwargs.get("target")
        if seed is not None:
            self._rng = random.Random(seed)

        if difficulty is None:
            difficulty = pick_difficulty(training_step, self._config, self._rng)

        # Resolve target: explicit kwarg wins; otherwise default Stage 1 (DRD2).
        # If the host has Stage 2 docking enabled, the full-name resolver picks up
        # the actually-active oracle so observations stay honest.
        if target is None:
            self._target_short = DEFAULT_TARGET
            full_name = get_active_target_name()
        else:
            if target not in KNOWN_TARGETS:
                raise ValueError(
                    f"unknown target {target!r}; must be one of {KNOWN_TARGETS} "
                    f"(or omit to use the default {DEFAULT_TARGET})."
                )
            self._target_short = target
            full_name = get_target_full_name(target)

        max_steps = max_steps_for(difficulty, self._config)

        starting = sample_starting_molecule(difficulty, self._rng)
        canonical = canonicalize_smiles(starting) or starting
        selfies = smiles_to_selfies(canonical) or "[C]"

        eid = episode_id or str(uuid4())
        self._state = MoleculeState(
            episode_id=eid,
            step_count=0,
            target=full_name,
            difficulty=difficulty,
            max_steps=max_steps,
            smiles=canonical,
            selfies=selfies,
            starting_smiles=canonical,
            edit_history=[],
            cumulative_reward=0.0,
            final_oracle_scores=None,
        )
        self._edit_history_full = []
        self._final_oracle_scores = None

        return self._build_observation(
            reward=0.0,
            done=False,
            last_action_valid=True,
            message=(
                f"Episode {eid[:8]} started @ {difficulty}. "
                f"Scaffold = {canonical}. Optimizing binding to {full_name}."
            ),
            truncated=False,
        )

    def step(self, action: MoleculeAction, **kwargs: Any) -> MoleculeObservation:
        """Apply one molecular edit. Returns observation with reward + done."""
        self._state.step_count += 1
        action_type = action.action_type

        # ─── TERMINATE ────────────────────────────────────────────────────
        if action_type == "TERMINATE":
            if self._state.step_count <= 1:
                # Cannot terminate immediately on step 1 — agent never edited
                reward = -0.5
                self._state.cumulative_reward += reward
                return self._build_observation(
                    reward=reward,
                    done=False,
                    last_action_valid=False,
                    message="Cannot TERMINATE on step 1 — make at least one edit first.",
                    truncated=False,
                )

            tr = terminal_reward(
                self._state.smiles,
                components_active=reward_components_for(self._state.difficulty, self._config),
                target=self._target_short,
            )
            self._final_oracle_scores = tr.components
            self._state.final_oracle_scores = tr.components
            self._state.cumulative_reward += tr.reward
            return self._build_observation(
                reward=tr.reward,
                done=True,
                last_action_valid=True,
                message=(
                    f"TERMINATED. Target={self._target_short} "
                    f"composite={tr.composite:.3f}, "
                    f"Lipinski={'PASS' if tr.lipinski_passes else 'FAIL'}, "
                    f"final SMILES={self._state.smiles}"
                ),
                truncated=False,
            )

        # ─── Mutation actions ─────────────────────────────────────────────
        last_action_valid = True
        message = ""
        smiles_before = self._state.smiles

        try:
            if action_type == "ADD_FRAGMENT":
                fragment = action.fragment
                if fragment is None:
                    raise MutationError("ADD_FRAGMENT requires 'fragment' field")
                vocab = get_vocab_for_difficulty(self._state.difficulty)
                if fragment not in vocab:
                    raise MutationError(f"fragment {fragment!r} not in {self._state.difficulty} vocab")
                new_smiles = apply_add_fragment(self._state.smiles, fragment, position=action.position)
                message = f"ADD_FRAGMENT({fragment}) → {new_smiles}"

            elif action_type == "REMOVE_FRAGMENT":
                if action.position is None:
                    raise MutationError("REMOVE_FRAGMENT requires 'position'")
                new_smiles = apply_remove_fragment(self._state.smiles, int(action.position))
                message = f"REMOVE@{action.position} → {new_smiles}"

            elif action_type == "SUBSTITUTE_ATOM":
                if action.position is None or action.new_atom is None:
                    raise MutationError("SUBSTITUTE_ATOM requires 'position' and 'new_atom'")
                new_smiles = apply_substitute_atom(self._state.smiles, int(action.position), str(action.new_atom))
                message = f"SUBSTITUTE@{action.position}→{action.new_atom} → {new_smiles}"

            else:
                raise MutationError(f"unknown action_type {action_type!r}")

            self._state.smiles = new_smiles
            self._state.selfies = smiles_to_selfies(new_smiles) or self._state.selfies
            edit_record = {"action": action_type, "before": smiles_before, "after": new_smiles}
            self._edit_history_full.append(edit_record)
            self._state.edit_history = list(self._edit_history_full[-5:])

        except MutationError as e:
            last_action_valid = False
            message = f"{action_type} rejected: {e}"

        # ─── Compute step shaping reward ─────────────────────────────────
        sr = step_shaping_reward(self._state.smiles, action_was_valid=last_action_valid)
        reward = sr.reward
        self._state.cumulative_reward += reward

        # ─── Truncation ──────────────────────────────────────────────────
        truncated = self._state.step_count >= self._state.max_steps
        done = truncated
        if truncated:
            tr = terminal_reward(
                self._state.smiles,
                components_active=reward_components_for(self._state.difficulty, self._config),
                target=self._target_short,
            )
            self._final_oracle_scores = tr.components
            self._state.final_oracle_scores = tr.components
            reward += tr.reward
            self._state.cumulative_reward += tr.reward
            message += f" [TRUNCATED — auto-terminal composite={tr.composite:.3f}]"

        return self._build_observation(
            reward=reward,
            done=done,
            last_action_valid=last_action_valid,
            message=message,
            truncated=truncated,
        )

    # ─── State property (OpenEnv interface contract) ────────────────────

    @property
    def state(self) -> MoleculeState:
        return deepcopy(self._state)

    # ─── Helpers ────────────────────────────────────────────────────────

    def _valid_actions_for_step(self) -> List[str]:
        base = ["ADD_FRAGMENT", "REMOVE_FRAGMENT", "SUBSTITUTE_ATOM"]
        if self._state.step_count >= 1:
            base.append("TERMINATE")
        return base

    def _properties_dict(self) -> dict:
        lipinski = check_lipinski(self._state.smiles)
        if lipinski is None:
            return {}
        return {
            "mw": float(lipinski.mw),
            "logp": float(lipinski.logp),
            "hbd": float(lipinski.hbd),
            "hba": float(lipinski.hba),
            "lipinski_violations": float(lipinski.violations),
        }

    def _build_observation(
        self,
        reward: float,
        done: bool,
        last_action_valid: bool,
        message: str,
        truncated: bool,
    ) -> MoleculeObservation:
        return MoleculeObservation(
            smiles=self._state.smiles,
            selfies=self._state.selfies,
            target=self._state.target,
            difficulty=self._state.difficulty,
            properties=self._properties_dict(),
            valid_actions=self._valid_actions_for_step(),
            available_fragments=get_vocab_for_difficulty(self._state.difficulty),
            steps_remaining=max(0, self._state.max_steps - self._state.step_count),
            last_action_valid=last_action_valid,
            message=message,
            truncated=truncated,
            done=done,
            reward=float(reward),
            metadata={
                "cumulative_reward": self._state.cumulative_reward,
                "final_oracle_scores": self._final_oracle_scores,
            },
        )
