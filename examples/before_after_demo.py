"""Generate the before/after demo molecules for the pitch video.

Runs N episodes with a base model and N with a trained model (or two LLM
endpoints). Saves a side-by-side comparison: starting molecule, base model
final, trained model final, with composite scores.

Usage:
    # Local server, default models
    python -m examples.before_after_demo --episodes 5

    # Compare specific endpoints
    python -m examples.before_after_demo \\
        --base-url http://localhost:8000 \\
        --base-model gpt-4o-mini \\
        --trained-model anshumanatrey/pharmarl-qwen-trained \\
        --episodes 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

from openai import OpenAI

from inference import run_episode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("before_after")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--base-model", default="gpt-4o-mini",
                   help="LLM endpoint for the BASELINE (untrained) agent")
    p.add_argument("--trained-model", default=None,
                   help="LLM endpoint for the TRAINED agent. If None, just runs baseline.")
    p.add_argument("--difficulty", default="trivial", choices=("trivial", "easy", "hard"))
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--out", default="demo_results.json")
    args = p.parse_args(argv)

    client = OpenAI()

    results: Dict[str, Any] = {"base": [], "trained": []}

    logger.info("=== BASELINE: %d episodes @ %s ===", args.episodes, args.difficulty)
    for i in range(args.episodes):
        s = run_episode(client, args.base_model, args.base_url, args.difficulty)
        results["base"].append({
            "final_smiles": s["final_smiles"],
            "cumulative_reward": s["cumulative_reward"],
        })
        logger.info("  ep%d: %.3f → %s", i + 1, s["cumulative_reward"], s["final_smiles"])

    if args.trained_model:
        # Use the trained model — point OpenAI client at a custom endpoint
        # if the trained model is served via vLLM/TGI, set OPENAI_BASE_URL env var.
        logger.info("=== TRAINED: %d episodes @ %s ===", args.episodes, args.difficulty)
        for i in range(args.episodes):
            s = run_episode(client, args.trained_model, args.base_url, args.difficulty)
            results["trained"].append({
                "final_smiles": s["final_smiles"],
                "cumulative_reward": s["cumulative_reward"],
            })
            logger.info("  ep%d: %.3f → %s", i + 1, s["cumulative_reward"], s["final_smiles"])

    base_avg = sum(r["cumulative_reward"] for r in results["base"]) / max(1, len(results["base"]))
    trained_avg = (
        sum(r["cumulative_reward"] for r in results["trained"]) / len(results["trained"])
        if results["trained"]
        else None
    )

    print("\n=== Summary ===")
    print(f"Baseline avg reward: {base_avg:.3f}")
    if trained_avg is not None:
        print(f"Trained  avg reward: {trained_avg:.3f}")
        print(f"Improvement:         {trained_avg - base_avg:+.3f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
