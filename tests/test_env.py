"""Env contract tests — reset/step state machine and reward logic."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MoleculeAction
from server.drug_discovery_environment import DrugDiscoveryEnvironment


@pytest.fixture
def env() -> DrugDiscoveryEnvironment:
    return DrugDiscoveryEnvironment(seed=42)


def test_reset_returns_observation_with_smiles(env: DrugDiscoveryEnvironment) -> None:
    obs = env.reset(difficulty="trivial")
    assert obs.smiles
    assert obs.difficulty == "trivial"
    assert obs.done is False


def test_step_advances_step_count(env: DrugDiscoveryEnvironment) -> None:
    env.reset(difficulty="trivial")
    obs = env.step(MoleculeAction(action_type="ADD_FRAGMENT", fragment="C", position=0))
    assert env.state.step_count == 1


def test_invalid_fragment_marked_invalid(env: DrugDiscoveryEnvironment) -> None:
    env.reset(difficulty="trivial")
    obs = env.step(MoleculeAction(action_type="ADD_FRAGMENT", fragment="NOT_IN_VOCAB", position=0))
    assert obs.last_action_valid is False
    assert env.state.step_count == 1


def test_terminate_too_early_penalized(env: DrugDiscoveryEnvironment) -> None:
    env.reset(difficulty="trivial")
    obs = env.step(MoleculeAction(action_type="TERMINATE"))
    assert obs.done is False
    assert obs.reward < 0


def test_terminate_after_edit_completes(env: DrugDiscoveryEnvironment) -> None:
    env.reset(difficulty="trivial")
    env.step(MoleculeAction(action_type="ADD_FRAGMENT", fragment="C", position=0))
    obs = env.step(MoleculeAction(action_type="TERMINATE"))
    assert obs.done is True
    assert "final_oracle_scores" in obs.metadata


def test_max_steps_truncates_episode(env: DrugDiscoveryEnvironment) -> None:
    env.reset(difficulty="trivial")  # max_steps=10
    last = None
    for _ in range(15):
        last = env.step(MoleculeAction(action_type="ADD_FRAGMENT", fragment="C", position=0))
        if last.done:
            break
    assert last is not None and last.done is True


def test_state_property_returns_valid_state(env: DrugDiscoveryEnvironment) -> None:
    env.reset(difficulty="trivial")
    state = env.state
    assert state.episode_id is not None
    assert state.target == "SARS-CoV-2_Mpro"
