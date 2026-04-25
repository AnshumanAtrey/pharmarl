"""Run one episode by hand — sanity-check the env without an LLM in the loop.

Usage:
    python -m examples.manual_episode

Walks through a hand-crafted sequence of edits and prints reward + state
after each step. Use this to verify the env logic before plugging in GRPO.
"""

from __future__ import annotations

import logging
import sys

import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MoleculeAction
from server.drug_discovery_environment import DrugDiscoveryEnvironment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main() -> int:
    env = DrugDiscoveryEnvironment(seed=42)
    obs = env.reset(difficulty="trivial")
    print(f"\nRESET → {obs.smiles}")
    print(f"  difficulty: {obs.difficulty}")
    print(f"  fragments:  {obs.available_fragments}")

    edits = [
        MoleculeAction(action_type="ADD_FRAGMENT", fragment="c1ccccc1", position=0),
        MoleculeAction(action_type="ADD_FRAGMENT", fragment="O", position=0),
        MoleculeAction(action_type="ADD_FRAGMENT", fragment="C(=O)O", position=0),
        MoleculeAction(action_type="TERMINATE"),
    ]
    for i, action in enumerate(edits, 1):
        obs = env.step(action)
        print(
            f"\nSTEP {i} action={action.action_type:<14s} "
            f"reward={obs.reward:+.3f} done={obs.done}"
        )
        print(f"  SMILES: {obs.smiles}")
        print(f"  msg:    {obs.message}")
        if obs.done:
            print(f"\n=== Episode done ===")
            print(f"  cumulative_reward: {obs.metadata.get('cumulative_reward', 0):.3f}")
            print(f"  final scores:      {obs.metadata.get('final_oracle_scores')}")
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
