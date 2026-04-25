"""§19-format demo: baseline vs trained, with reward breakdown + safeguards.

This is the script you run to produce the numbers (and JSON) for the pitch
video and the README "Reward improvement" table. It is intentionally
self-contained — no LLM call required for the baseline, so judges can
reproduce the demo by running this against a deployed env even without
GPU access.

Three policies:
  random       — pick uniformly from valid actions. The "untrained" floor.
  model        — call a HF transformers model (LoRA-loaded) for actions.
                 Used for the BEFORE (base model) and AFTER (trained) runs.
  scripted     — a hand-coded rule of thumb that always extends with one
                 functional fragment. Useful sanity ceiling.

Usage:
    # Quickest sanity check — random vs scripted, no GPU:
    python -m examples.demo --base-url http://localhost:8000 \\
        --episodes 5 --policies random scripted

    # Real demo — base model vs trained LoRA:
    python -m examples.demo --base-url https://YOUR-SPACE.hf.space \\
        --episodes 10 --policies model:base model:trained \\
        --base-model-id unsloth/Llama-3.2-1B-Instruct \\
        --trained-model-id YOUR-USER/pharmarl-llama-trained \\
        --out demo_results
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests


# ─── Policies ────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_action_json(text: str) -> Optional[dict]:
    for m in _JSON_RE.findall(text):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue
    return None


def random_policy(obs: dict, rng: random.Random) -> dict:
    """Uniform over valid actions. The 'no skill' floor."""
    valid = obs["valid_actions"]
    fragments = obs["available_fragments"]
    action_type = rng.choice(valid)
    if action_type == "ADD_FRAGMENT":
        return {
            "action_type": "ADD_FRAGMENT",
            "fragment": rng.choice(fragments),
            "position": 0,
        }
    if action_type == "REMOVE_FRAGMENT":
        return {"action_type": "REMOVE_FRAGMENT", "position": 0}
    if action_type == "SUBSTITUTE_ATOM":
        return {
            "action_type": "SUBSTITUTE_ATOM",
            "position": 0,
            "new_atom": rng.choice(["F", "N", "O", "Cl"]),
        }
    return {"action_type": "TERMINATE"}


def scripted_policy(obs: dict, rng: random.Random) -> dict:
    """Always add a fragment; terminate at the budget. Hand-coded ceiling."""
    if obs["steps_remaining"] <= 1 and "TERMINATE" in obs["valid_actions"]:
        return {"action_type": "TERMINATE"}
    fragments = obs["available_fragments"]
    drug_like = [f for f in fragments if any(c in f for c in "NOFcn")]
    pick = rng.choice(drug_like or fragments)
    return {"action_type": "ADD_FRAGMENT", "fragment": pick, "position": 0}


def make_model_policy(model_id: str, system_prompt: str):
    """Load a HF transformers model + tokenizer; return a policy fn."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    def policy(obs: dict, rng: random.Random) -> dict:
        prompt = (
            f"{system_prompt}\n\nSMILES: {obs['smiles']}\n"
            f"Fragments: {obs['available_fragments'][:8]}\n"
            f"Valid actions: {obs['valid_actions']}\n"
            f"Respond with JSON action:"
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=80, do_sample=True,
                temperature=0.7, top_p=0.95, pad_token_id=tok.eos_token_id,
            )
        txt = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        action = parse_action_json(txt)
        if action is None:
            return random_policy(obs, rng)
        return action

    return policy


# ─── Episode runner ──────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    starting_smiles: str = ""
    final_smiles: str = ""
    cumulative_reward: float = 0.0
    n_steps: int = 0
    n_invalid: int = 0
    final_components: Dict[str, float] = field(default_factory=dict)
    lipinski_passes: bool = False


def run_episode(
    base_url: str,
    policy: Callable[[dict, random.Random], dict],
    rng: random.Random,
    difficulty: str = "trivial",
) -> EpisodeResult:
    r = requests.post(f"{base_url}/reset", json={"difficulty": difficulty}).json()
    obs = r["observation"]
    res = EpisodeResult(starting_smiles=obs["smiles"])
    cumulative = 0.0
    invalid = 0
    while True:
        action = policy(obs, rng)
        step = requests.post(f"{base_url}/step", json=action).json()
        cumulative += step["reward"]
        if not step["observation"].get("last_action_valid", True):
            invalid += 1
        obs = step["observation"]
        res.n_steps += 1
        if step["done"]:
            md = step.get("metadata", {}) or step.get("observation", {}).get("metadata", {})
            res.final_components = (md or {}).get("final_oracle_scores") or {}
            br = (md or {}).get("reward_breakdown") or {}
            res.lipinski_passes = bool(br.get("lipinski_passes", False))
            break
    res.final_smiles = obs["smiles"]
    res.cumulative_reward = cumulative
    res.n_invalid = invalid
    return res


# ─── Reporting ───────────────────────────────────────────────────────────

def summarize(label: str, runs: List[EpisodeResult]) -> dict:
    cums = [r.cumulative_reward for r in runs]
    qed = [r.final_components.get("qed", 0.0) for r in runs]
    docking = [r.final_components.get("docking", 0.0) for r in runs]
    lip = [r.lipinski_passes for r in runs]
    return {
        "label": label,
        "n": len(runs),
        "reward_mean": statistics.mean(cums) if cums else 0.0,
        "reward_max": max(cums) if cums else 0.0,
        "reward_min": min(cums) if cums else 0.0,
        "reward_std": statistics.pstdev(cums) if len(cums) > 1 else 0.0,
        "qed_mean": statistics.mean(qed) if qed else 0.0,
        "docking_mean": statistics.mean(docking) if docking else 0.0,
        "lipinski_pass_rate": sum(lip) / len(lip) if lip else 0.0,
        "invalid_action_rate": sum(r.n_invalid for r in runs) / max(1, sum(r.n_steps for r in runs)),
    }


def render_markdown(summaries: List[dict], runs_by_label: Dict[str, List[EpisodeResult]]) -> str:
    lines = []
    lines.append("# PharmaRL Demo Results\n")
    lines.append("## Summary\n")
    lines.append("| Policy | N | Mean Reward | Max | QED | Docking | Lipinski Pass | Invalid Action |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s['label']} | {s['n']} | {s['reward_mean']:+.3f} | {s['reward_max']:+.3f} | "
            f"{s['qed_mean']:.3f} | {s['docking_mean']:.3f} | "
            f"{s['lipinski_pass_rate']:.0%} | {s['invalid_action_rate']:.0%} |"
        )
    if len(summaries) >= 2:
        a, b = summaries[0], summaries[-1]
        delta = b["reward_mean"] - a["reward_mean"]
        rel = (b["reward_mean"] - a["reward_mean"]) / abs(a["reward_mean"]) * 100 if a["reward_mean"] else 0
        lines.append(f"\n**Improvement {a['label']} → {b['label']}: {delta:+.3f} ({rel:+.0f}%)**\n")

    lines.append("\n## Sample Trajectories (final molecule per episode)\n")
    for label, runs in runs_by_label.items():
        lines.append(f"### {label}")
        for i, r in enumerate(runs[:5]):
            lip = "✅" if r.lipinski_passes else "❌"
            lines.append(
                f"- ep{i+1}: `{r.starting_smiles}` → `{r.final_smiles}`  "
                f"reward={r.cumulative_reward:+.3f}  Lipinski={lip}"
            )
        lines.append("")

    lines.append("## Safeguards demonstrated\n")
    all_runs = [r for runs in runs_by_label.values() for r in runs]
    lipinski_failed = [r for r in all_runs if not r.lipinski_passes]
    if lipinski_failed:
        avg_failed = statistics.mean(r.cumulative_reward for r in lipinski_failed)
        passed = [r for r in all_runs if r.lipinski_passes]
        avg_passed = statistics.mean(r.cumulative_reward for r in passed) if passed else 0.0
        lines.append(
            f"- **Lipinski gate active**: {len(lipinski_failed)} molecule(s) failed Rule of 5 "
            f"and were penalized (avg reward {avg_failed:+.2f} vs {avg_passed:+.2f} for passers)."
        )
    invalid_total = sum(r.n_invalid for r in all_runs)
    if invalid_total:
        lines.append(
            f"- **Invalid-action penalty**: {invalid_total} malformed actions "
            f"caught by SELFIES/RDKit validity, penalized -0.1 each."
        )
    lines.append(
        "- **Verifiable rewards only**: every score above came from TDC oracles "
        "or RDKit — no LLM judge in the reward path."
    )
    return "\n".join(lines)


# ─── CLI ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You design SARS-CoV-2 Mpro inhibitors by editing SMILES molecules. "
    "Respond with ONE JSON action per turn. Allowed: ADD_FRAGMENT, "
    "REMOVE_FRAGMENT, SUBSTITUTE_ATOM, TERMINATE."
)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--difficulty", default="trivial", choices=("trivial", "easy", "hard"))
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="demo_results")
    p.add_argument(
        "--policies", nargs="+", default=["random", "scripted"],
        help="Names: random, scripted, or model:<label>",
    )
    p.add_argument("--base-model-id", default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--trained-model-id", default=None,
                   help="HF Hub id of trained LoRA model (required if 'model:trained' in --policies)")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    summaries: List[dict] = []
    runs_by_label: Dict[str, List[EpisodeResult]] = {}

    for spec in args.policies:
        if spec == "random":
            label = "random"
            policy = lambda o, r: random_policy(o, r)  # noqa: E731
        elif spec == "scripted":
            label = "scripted"
            policy = lambda o, r: scripted_policy(o, r)  # noqa: E731
        elif spec.startswith("model:"):
            tag = spec.split(":", 1)[1]
            mid = args.trained_model_id if tag == "trained" else args.base_model_id
            if mid is None:
                print(f"ERROR: --trained-model-id required for {spec}", file=sys.stderr)
                return 2
            label = f"model:{tag} ({mid})"
            print(f"Loading {label} ...", file=sys.stderr)
            policy = make_model_policy(mid, SYSTEM_PROMPT)
        else:
            print(f"Unknown policy spec: {spec}", file=sys.stderr)
            return 2

        runs = []
        t0 = time.time()
        for i in range(args.episodes):
            res = run_episode(args.base_url, policy, rng, args.difficulty)
            runs.append(res)
            print(
                f"  [{label}] ep{i+1:2d}: R={res.cumulative_reward:+.2f}  "
                f"steps={res.n_steps}  final={res.final_smiles}",
                file=sys.stderr,
            )
        dur = time.time() - t0
        print(f"  → {label} finished {args.episodes} eps in {dur:.1f}s", file=sys.stderr)

        runs_by_label[label] = runs
        summaries.append(summarize(label, runs))

    # Save JSON
    json_payload = {
        "config": vars(args),
        "summaries": summaries,
        "runs": {
            label: [
                {
                    "starting_smiles": r.starting_smiles,
                    "final_smiles": r.final_smiles,
                    "cumulative_reward": r.cumulative_reward,
                    "n_steps": r.n_steps,
                    "n_invalid": r.n_invalid,
                    "lipinski_passes": r.lipinski_passes,
                    "final_components": r.final_components,
                }
                for r in runs
            ]
            for label, runs in runs_by_label.items()
        },
    }
    (out_path / "results.json").write_text(json.dumps(json_payload, indent=2))
    md = render_markdown(summaries, runs_by_label)
    (out_path / "results.md").write_text(md)
    print(f"\nWrote {out_path / 'results.json'}", file=sys.stderr)
    print(f"Wrote {out_path / 'results.md'}", file=sys.stderr)
    print("\n" + md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
