"""Eval a policy with statistical confidence intervals.

Runs N episodes per target × M targets and reports mean ± std for each.
Output is JSON in a standardized format that `plot_results.py` consumes
to generate the README plots.

Usage examples:
    # Untrained Qwen baseline on 3 targets, 10 eps each
    python -m examples.eval_with_ci --policy random --episodes 10 \
        --targets DRD2 GSK3B JNK3 --output runs/random_baseline.json

    # Scripted policy
    python -m examples.eval_with_ci --policy scripted --episodes 10 \
        --targets DRD2 GSK3B JNK3 --output runs/scripted.json

    # Use against the live HF Space
    PHARMARL_ENV_URL=https://anshumanatrey-pharmarl.hf.space \
        python -m examples.eval_with_ci --policy random --episodes 10 \
        --targets DRD2 GSK3B JNK3 --output runs/random_live.json

The output JSON shape:
    {
        "policy": "random",
        "env_url": "...",
        "results": {
            "DRD2":  {"episodes": [...], "mean": 2.78, "std": 0.41, "n": 10},
            "GSK3B": {"episodes": [...], "mean": 2.33, "std": 0.62, "n": 10},
            "JNK3":  {"episodes": [...], "mean": 1.78, "std": 0.55, "n": 10}
        },
        "overall": {"mean": 2.30, "std": 0.50}
    }
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List

import requests

# Reuse the policies from examples/demo.py — keep one source of truth.
from examples.demo import POLICIES, run_episode  # type: ignore


def eval_policy_on_target(
    session: requests.Session,
    env_url: str,
    policy_name: str,
    target: str,
    episodes: int,
    difficulty: str = "easy",
    seed_base: int = 0,
) -> Dict[str, Any]:
    """Run `episodes` rollouts and report per-episode + summary statistics."""
    cumulatives = []
    final_smiles = []
    for i in range(episodes):
        result = run_episode(
            session=session,
            env_url=env_url,
            policy_name=policy_name,
            target=target,
            difficulty=difficulty,
            max_steps=20,
        )
        cumulatives.append(result["cumulative"])
        final_smiles.append(result["final_smiles"])

    mean = statistics.mean(cumulatives)
    std = statistics.stdev(cumulatives) if len(cumulatives) > 1 else 0.0
    return {
        "episodes": cumulatives,
        "final_smiles": final_smiles,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "n": len(cumulatives),
        "min": round(min(cumulatives), 4),
        "max": round(max(cumulatives), 4),
    }


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True, choices=list(POLICIES.keys()) + ["trained_qwen"],
                   help="Policy to evaluate. 'trained_qwen' is a placeholder for an external runner.")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--targets", nargs="+", default=["DRD2", "GSK3B", "JNK3"])
    p.add_argument("--difficulty", default="easy", choices=("trivial", "easy", "hard"))
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--output", required=True, help="Path to write eval JSON")
    p.add_argument("--env-url", default=os.environ.get("PHARMARL_ENV_URL", "http://127.0.0.1:8000"))
    args = p.parse_args(argv)

    if args.policy == "trained_qwen":
        print("[ERROR] --policy trained_qwen requires the trained-model runner",
              "in colab/eval_trained.py — not implemented in this script.",
              "Use the dedicated trained-model eval cell in the Colab notebook.",
              file=sys.stderr)
        return 2

    session = requests.Session()
    health = session.get(f"{args.env_url}/health", timeout=10)
    if health.status_code != 200:
        print(f"[FAIL] env health check: {health.status_code}", file=sys.stderr)
        return 1

    print(f"Eval: policy={args.policy} episodes={args.episodes}/target targets={args.targets}")
    out: Dict[str, Any] = {
        "policy": args.policy,
        "env_url": args.env_url,
        "difficulty": args.difficulty,
        "results": {},
    }

    all_cumulatives: List[float] = []
    for target in args.targets:
        print(f"\n=== {target} ===")
        per_target = eval_policy_on_target(
            session=session,
            env_url=args.env_url,
            policy_name=args.policy,
            target=target,
            episodes=args.episodes,
            difficulty=args.difficulty,
            seed_base=args.seed_base,
        )
        out["results"][target] = per_target
        all_cumulatives.extend(per_target["episodes"])
        print(f"  mean={per_target['mean']:+.3f}  std={per_target['std']:.3f}  "
              f"n={per_target['n']}  range=[{per_target['min']:+.3f}, {per_target['max']:+.3f}]")

    overall_mean = statistics.mean(all_cumulatives)
    overall_std = statistics.stdev(all_cumulatives) if len(all_cumulatives) > 1 else 0.0
    out["overall"] = {
        "mean": round(overall_mean, 4),
        "std": round(overall_std, 4),
        "n": len(all_cumulatives),
    }
    print(f"\nOVERALL: mean={overall_mean:+.3f} std={overall_std:.3f} n={len(all_cumulatives)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
