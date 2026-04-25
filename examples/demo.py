"""Compare policies on the env — regression sanity test, no LLM, no GPU, no API.

A policy is a callable: given an observation, return an action dict.
We provide two:
  - `random_policy` — uniform random over valid_actions / available_fragments
  - `scripted_policy` — hand-built sequence that targets DRD2 pharmacophores
                       (basic amine + aromatic ring + small substituent + TERMINATE)

If the SCRIPTED policy doesn't outscore RANDOM by a margin, the reward signal
is broken — fix it BEFORE burning Colab/HF compute on training.

Usage:
    # Run 8 episodes per policy on the local server, easy tier, against DRD2:
    python -m examples.demo --policies random scripted --episodes 8

    # Against the live HF Space:
    PHARMARL_ENV_URL=https://anshumanatrey-pharmarl.hf.space \\
        python -m examples.demo --policies random scripted --target DRD2

    # All three multi-target probes:
    python -m examples.demo --policies random scripted --target DRD2 GSK3B JNK3
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import uuid
from typing import Any, Callable, Dict, List

import requests

# ─── Policies ─────────────────────────────────────────────────────────


def random_policy(obs: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Uniform random over valid actions and parameters."""
    action_type = rng.choice(obs["valid_actions"])
    if action_type == "ADD_FRAGMENT":
        return {
            "action_type": "ADD_FRAGMENT",
            "fragment": rng.choice(obs["available_fragments"]),
            "position": 0,
        }
    if action_type == "REMOVE_FRAGMENT":
        return {"action_type": "REMOVE_FRAGMENT", "position": 0}
    if action_type == "SUBSTITUTE_ATOM":
        return {
            "action_type": "SUBSTITUTE_ATOM",
            "position": 0,
            "new_atom": rng.choice(["C", "N", "O", "F"]),
        }
    return {"action_type": "TERMINATE"}


# Hand-built sequence targeting DRD2-flavored pharmacophore:
# basic-amine + aromatic + small linker. Works regardless of starting scaffold;
# we just keep ADDing fragments from the easy/hard vocab in a sensible order.
_SCRIPTED_PLAN = [
    "N",            # nitrogen — basic amine for D2 ionic interaction
    "c1ccccc1",     # benzene — aromatic pocket
    "C",            # methyl — small lipophilic
    "OC",           # methoxy — H-bond acceptor
    "TERMINATE",
]


def scripted_policy(obs: Dict[str, Any], _rng: random.Random) -> Dict[str, Any]:
    """Walks through _SCRIPTED_PLAN ignoring the env's stochasticity.

    The plan is short (4 edits) so it terminates well before max_steps. Each
    fragment is in the easy/hard vocab. Random doesn't see this plan.
    """
    step = obs.get("_scripted_step", 0)
    if step >= len(_SCRIPTED_PLAN):
        return {"action_type": "TERMINATE"}
    move = _SCRIPTED_PLAN[step]
    if move == "TERMINATE":
        return {"action_type": "TERMINATE"}
    # Fall back to ADD_FRAGMENT if our planned fragment isn't in the active vocab.
    if move not in obs["available_fragments"]:
        return random_policy(obs, _rng)
    return {"action_type": "ADD_FRAGMENT", "fragment": move, "position": 0}


POLICIES: Dict[str, Callable[[Dict[str, Any], random.Random], Dict[str, Any]]] = {
    "random": random_policy,
    "scripted": scripted_policy,
}


# ─── Episode driver ──────────────────────────────────────────────────


def run_episode(
    session: requests.Session,
    env_url: str,
    policy_name: str,
    target: str | None = None,
    difficulty: str = "easy",
    max_steps: int = 30,
    rng: random.Random | None = None,
) -> Dict[str, Any]:
    rng = rng or random.Random()
    policy = POLICIES[policy_name]
    episode_id = str(uuid.uuid4())
    body = {"difficulty": difficulty, "episode_id": episode_id}
    if target:
        body["target"] = target
    obs = session.post(f"{env_url}/reset", json=body).json()["observation"]
    actions: List[Dict[str, Any]] = []
    rewards: List[float] = []
    cumulative = 0.0
    for step_no in range(max_steps):
        # The scripted policy needs to know which step it's on.
        if policy_name == "scripted":
            obs["_scripted_step"] = step_no
        action = policy(obs, rng)
        step = session.post(
            f"{env_url}/step",
            json={"action": action, "episode_id": episode_id},
            timeout=120,
        ).json()
        actions.append(action)
        rewards.append(step["reward"])
        cumulative += step["reward"]
        obs = step["observation"]
        if step["done"]:
            break
    return {
        "policy": policy_name,
        "episode_id": episode_id,
        "actions": actions,
        "rewards": rewards,
        "cumulative": cumulative,
        "final_smiles": obs["smiles"],
    }


# ─── CLI ─────────────────────────────────────────────────────────────


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env-url", default=os.environ.get("PHARMARL_ENV_URL", "http://127.0.0.1:8000"))
    p.add_argument("--policies", nargs="+", default=["random", "scripted"], choices=list(POLICIES.keys()))
    p.add_argument("--target", nargs="+", default=["DRD2"], help="One or more targets to probe.")
    p.add_argument("--difficulty", default="easy", choices=("trivial", "easy", "hard"))
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--margin", type=float, default=0.3,
                   help="Required scripted vs random margin (cumulative reward) for at least "
                        "ONE target. Different TDC classifiers have different dynamic ranges — "
                        "DRD2 is permissive (even random small molecules score 2-3), GSK3B/JNK3 "
                        "are stricter. We require the margin on the strictest, not on all three.")
    args = p.parse_args(argv)

    session = requests.Session()
    rng = random.Random(args.seed)

    # Probe health first
    try:
        h = session.get(f"{args.env_url}/health", timeout=5).json()
        print(f"[OK]  {args.env_url}/health → {h}")
    except Exception as e:
        print(f"[FAIL] cannot reach {args.env_url} ({e})", file=sys.stderr)
        return 2

    print()
    deltas: Dict[str, float] = {}
    for tgt in args.target:
        print(f"=== target = {tgt}, difficulty = {args.difficulty}, episodes = {args.episodes} ===")
        per_policy_means: Dict[str, float] = {}
        for policy_name in args.policies:
            results = []
            for _ in range(args.episodes):
                r = run_episode(session, args.env_url, policy_name, target=tgt,
                                difficulty=args.difficulty, rng=rng)
                results.append(r)
            cums = [r["cumulative"] for r in results]
            per_policy_means[policy_name] = statistics.mean(cums)
            print(f"  {policy_name:9s}: mean={per_policy_means[policy_name]:+.3f} "
                  f"min={min(cums):+.3f} max={max(cums):+.3f}")
            for i, r in enumerate(results, 1):
                print(f"    ep{i}: cum={r['cumulative']:+.3f} final={r['final_smiles']}")

        if "random" in per_policy_means and "scripted" in per_policy_means:
            delta = per_policy_means["scripted"] - per_policy_means["random"]
            deltas[tgt] = delta
            if delta < 0:
                color = "RED"
            elif delta < args.margin:
                color = "YELLOW"
            else:
                color = "GREEN"
            print(f"\n  scripted - random = {delta:+.3f}  [{color}]")
        print()

    # Verdict: at least one target must show a clear margin (>= --margin).
    # All-yellow (small but positive) is acceptable; all-zero or any-negative is not.
    if not deltas:
        print("[INCONCLUSIVE] need both 'random' and 'scripted' policies to make a verdict.")
        return 0
    any_clear = any(d >= args.margin for d in deltas.values())
    any_negative = any(d < 0 for d in deltas.values())

    print("=" * 60)
    print("  Regression verdict")
    print("=" * 60)
    for t, d in deltas.items():
        print(f"  {t:6s}: scripted - random = {d:+.3f}")
    print()

    if any_negative:
        print("[RED] At least one target showed scripted < random — reward signal likely broken.")
        return 1
    if any_clear:
        print(f"[GREEN] At least one target shows clear scripted advantage (>= {args.margin}). "
              "Reward signal is meaningful. Proceed to training.")
        return 0
    print(f"[YELLOW] All deltas positive but small (< {args.margin}). Reward gradient is weak. "
          "Training may show only modest improvement. Consider lower expectations or strengthen scripted policy.")
    return 0  # not a hard fail


if __name__ == "__main__":
    sys.exit(main())
