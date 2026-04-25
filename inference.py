"""Baseline agent loop for PharmaRL.

Runs an OpenAI-client-compatible LLM (or the base Qwen3 via vLLM/Ollama)
against the env. Useful for:
  - Smoke-testing the env end-to-end
  - Generating "before training" reference trajectories for the demo video
  - Hand-debugging prompts before the GRPO training run

Usage:
    python inference.py --base-url http://localhost:8000 --difficulty trivial
    python inference.py --base-url <hf-space-url> --difficulty hard --episodes 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import PharmaRLEnv
from models import MoleculeAction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("inference")


SYSTEM_PROMPT = """You are a medicinal chemist designing a small-molecule drug.

You will edit a molecule across multiple steps to optimize it for the target
reported in each observation's `target` field (Stage 1 default: DRD2 dopamine
D2 receptor — a CNS-therapeutic-relevant target). Each step you may:

  1. ADD_FRAGMENT — attach a fragment from the available list at a heavy-atom position
  2. REMOVE_FRAGMENT — remove the heavy atom at a position (molecule must stay connected)
  3. SUBSTITUTE_ATOM — replace an atom with C/N/O/S/F/Cl/Br/I/P
  4. TERMINATE — submit your current molecule for scoring (only after step 1)

You MUST respond with a JSON object on a single line, like:
  {"action_type": "ADD_FRAGMENT", "fragment": "c1ccccc1", "position": 0}
  {"action_type": "TERMINATE"}

Valid Lipinski-passing molecules (MW<=500, LogP<=5, HBD<=5, HBA<=10) score higher.
Higher QED, higher predicted binding, lower toxicity = better composite reward.
"""


def build_prompt(observation_dict: Dict[str, Any]) -> str:
    obs = observation_dict
    fragments_short = obs.get("available_fragments", [])[:15]
    return f"""Step {obs.get('step_count', '?')} / {obs.get('step_count', 0) + obs.get('steps_remaining', 0)}
Target: {obs.get('target')}
Difficulty: {obs.get('difficulty')}
Current SMILES: {obs.get('smiles')}
Properties: {json.dumps(obs.get('properties', {}), indent=None)}
Valid actions: {obs.get('valid_actions')}
Available fragments (sample): {fragments_short}

Last message: {obs.get('message')}

Respond with one JSON action."""


_JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_action(text: str) -> Optional[MoleculeAction]:
    """Pull the first valid JSON object out of the model output."""
    candidates = _JSON_RE.findall(text)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            return MoleculeAction(**payload)
        except Exception:
            continue
    return None


def run_episode(
    client: OpenAI,
    model: str,
    env_url: str,
    difficulty: str,
) -> Dict[str, Any]:
    transcript: List[Dict[str, Any]] = []
    with PharmaRLEnv(base_url=env_url).sync() as env:
        result = env.reset(difficulty=difficulty)
        observation = result.observation.model_dump()
        observation["step_count"] = 0
        cumulative = 0.0

        while True:
            user_prompt = build_prompt(observation)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=120,
                temperature=0.7,
            )
            text = response.choices[0].message.content or ""
            action = parse_action(text)
            transcript.append(
                {"prompt": user_prompt, "raw": text, "parsed": action and action.model_dump()}
            )

            if action is None:
                # Issue a parse-fail no-op so the env can record the penalty
                action = MoleculeAction(action_type="ADD_FRAGMENT", fragment="?", position=0)

            step_result = env.step(action)
            observation = step_result.observation.model_dump()
            observation["step_count"] = transcript[-1].get("step_count", 0) + 1
            cumulative += float(step_result.reward or 0.0)
            logger.info(
                "step=%d action=%s reward=%.3f cum=%.3f smiles=%s",
                len(transcript),
                action.action_type,
                step_result.reward or 0.0,
                cumulative,
                observation.get("smiles"),
            )
            if step_result.done:
                break

    return {
        "transcript": transcript,
        "cumulative_reward": cumulative,
        "final_smiles": observation.get("smiles"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--difficulty", default="trivial", choices=("trivial", "easy", "hard"))
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    args = p.parse_args(argv)

    client = OpenAI()
    summaries = []
    for i in range(args.episodes):
        logger.info("--- Episode %d ---", i + 1)
        summary = run_episode(client, args.model, args.base_url, args.difficulty)
        summaries.append(summary)
        logger.info(
            "Episode %d done: reward=%.3f final=%s",
            i + 1,
            summary["cumulative_reward"],
            summary["final_smiles"],
        )

    avg = sum(s["cumulative_reward"] for s in summaries) / max(1, len(summaries))
    logger.info("Avg cumulative reward across %d eps: %.3f", len(summaries), avg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
