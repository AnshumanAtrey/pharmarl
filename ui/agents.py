"""Agent strategies for the PharmaRL Gradio UI.

Each agent exposes `next_action(observation) -> MoleculeAction`. The UI
loops, feeds observations in, and renders the resulting trajectory.

Live agents (Random, Scripted) are in-process and instant. The Llama
agents are intentionally stub-only — calling a live LLM API per step
during a 3-min hackathon demo is too slow and brittle. Instead the UI
shows the published baselines table for those rows, and labels them as
"static reference". When the H200 adapter is ready we wire in a live
adapter call here.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

from models import MoleculeAction, MoleculeObservation


@dataclass
class AgentInfo:
    key: str
    label: str
    blurb: str
    live: bool       # True = the UI runs it; False = pre-computed baseline shown
    baseline: Optional[float] = None  # mean reward across DRD2/GSK3B/JNK3 (for non-live)


# Source: docs/baselines.md / README "Baseline spectrum" — reproduced so
# the UI can surface them without making live API calls.
AGENTS: list[AgentInfo] = [
    AgentInfo("random",   "🎲 Random",            "Uniform random action from the valid set. The 'do nothing learned' floor.", True),
    AgentInfo("scripted", "📜 Scripted",         "Hand-written 4-step recipe (aromatic + amine + carbonyl + terminate). Beats random by design.", True),
    AgentInfo("llama1b",  "🦙 Llama 3.2 1B",     "Untrained Llama 3.2 1B — out-of-the-box, no PharmaRL exposure.",        False, baseline=None),
    AgentInfo("llama3b",  "🦙 Llama 3.2 3B",     "Untrained 3B; published baseline (DRD2 +1.80 / GSK3B +1.99 / JNK3 +1.22).", False, baseline=1.67),
    AgentInfo("llama8b",  "🦙 Llama 3.1 8B",     "Best small-model baseline (+2.45). Beats Random and Scripted.",        False, baseline=2.45),
    AgentInfo("llama70b", "🦙 Llama 3.3 70B",    "Frontier baseline — and worse than random (+1.19). Inverted scaling proof.", False, baseline=1.19),
    AgentInfo("gemini_flash", "✨ Gemini 2.5 Flash", "Mid-tier Gemini baseline (+1.81).",                                False, baseline=1.81),
    AgentInfo("gemini_pro",   "✨ Gemini 2.5 Pro",   "Best baseline overall (+3.68). Only frontier model that clears the scripted floor.", False, baseline=3.68),
    AgentInfo("trained",      "🚀 Llama 1B trained (vijay-h200)", "Your trained adapter from the H200 GRPO run. Wired in after the run completes.", False, baseline=None),
]


def get_agent(key: str) -> AgentInfo:
    for a in AGENTS:
        if a.key == key:
            return a
    raise KeyError(key)


# ─── Live agents ─────────────────────────────────────────────────────────


class RandomAgent:
    """Uniform random action drawn from the env's valid_actions list and
    the available_fragments vocab. Seeded for reproducibility per UI run."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def next_action(self, obs: MoleculeObservation) -> MoleculeAction:
        valid = list(obs.valid_actions) or ["ADD_FRAGMENT"]
        action_type = self.rng.choice(valid)
        if action_type == "ADD_FRAGMENT":
            frag = self.rng.choice(obs.available_fragments or ["C"])
            pos = self.rng.randint(0, max(0, _atom_count(obs.smiles) - 1))
            return MoleculeAction(action_type="ADD_FRAGMENT", fragment=frag, position=pos)
        if action_type == "SUBSTITUTE_ATOM":
            atom = self.rng.choice(["F", "N", "O", "Cl", "S"])
            pos = self.rng.randint(0, max(0, _atom_count(obs.smiles) - 1))
            return MoleculeAction(action_type="SUBSTITUTE_ATOM", new_atom=atom, position=pos)
        if action_type == "REMOVE_FRAGMENT":
            pos = self.rng.randint(0, max(0, _atom_count(obs.smiles) - 1))
            return MoleculeAction(action_type="REMOVE_FRAGMENT", position=pos)
        return MoleculeAction(action_type="TERMINATE")


class ScriptedAgent:
    """Four-step recipe: nucleate aromatic + add amine + decorate + terminate.
    Mirrors the 'Scripted (4-step)' baseline in the README. Per-difficulty
    tweaks make sure it picks fragments that are actually in vocab."""

    PLAN_BY_DIFFICULTY: dict[str, List[dict]] = {
        "trivial": [
            {"action_type": "ADD_FRAGMENT", "fragment_pref": "c1ccncc1", "position": 0},
            {"action_type": "SUBSTITUTE_ATOM", "new_atom": "F", "position": 0},
            {"action_type": "ADD_FRAGMENT", "fragment_pref": "C(=O)O", "position": 0},
            {"action_type": "TERMINATE"},
        ],
        "easy": [
            {"action_type": "ADD_FRAGMENT", "fragment_pref": "c1ccc(N)cc1", "position": 0},
            {"action_type": "ADD_FRAGMENT", "fragment_pref": "C(=O)O", "position": 0},
            {"action_type": "SUBSTITUTE_ATOM", "new_atom": "F", "position": 1},
            {"action_type": "TERMINATE"},
        ],
        "hard": [
            {"action_type": "ADD_FRAGMENT", "fragment_pref": "c1ccc(N)cc1", "position": 0},
            {"action_type": "ADD_FRAGMENT", "fragment_pref": "N1CCNCC1", "position": 0},
            {"action_type": "ADD_FRAGMENT", "fragment_pref": "C(=O)O", "position": 0},
            {"action_type": "SUBSTITUTE_ATOM", "new_atom": "F", "position": 1},
            {"action_type": "TERMINATE"},
        ],
    }

    def __init__(self, difficulty: str = "trivial"):
        self.plan = list(self.PLAN_BY_DIFFICULTY.get(difficulty, self.PLAN_BY_DIFFICULTY["trivial"]))
        self.idx = 0

    def next_action(self, obs: MoleculeObservation) -> MoleculeAction:
        if self.idx >= len(self.plan):
            return MoleculeAction(action_type="TERMINATE")
        step = dict(self.plan[self.idx])
        self.idx += 1
        # Resolve fragment_pref against the current vocab so we don't 422 on
        # an out-of-vocab choice; fall back to whatever the env offers.
        if step.get("action_type") == "ADD_FRAGMENT":
            pref = step.pop("fragment_pref", None)
            vocab = obs.available_fragments or [pref]
            chosen = pref if pref in vocab else (vocab[0] if vocab else "C")
            step["fragment"] = chosen
        action_type = step.pop("action_type")
        return MoleculeAction(action_type=action_type, **step)


def _atom_count(smiles: str) -> int:
    if not smiles:
        return 1
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smiles)
        return m.GetNumAtoms() if m else 1
    except Exception:
        return 1


def make_agent(key: str, difficulty: str):
    """Factory used by the UI. Returns the live agent instance, or None
    for non-live (baseline-only) agents."""
    if key == "random":
        return RandomAgent()
    if key == "scripted":
        return ScriptedAgent(difficulty)
    return None
