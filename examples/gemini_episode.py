"""Drive PharmaRL via Gemini — proves the env works with a real LLM as the policy.

Reads GEMINI_API_KEY and PHARMARL_ENV_URL from .env (gitignored).

Usage:
    # Default: 3 episodes × DRD2/GSK3B/JNK3, gemini-2.5-flash, against the live HF Space
    python -m examples.gemini_episode

    # Pricier model:
    python -m examples.gemini_episode --model gemini-2.5-pro --episodes 2

    # Single target probe:
    python -m examples.gemini_episode --target DRD2 --episodes 5

    # Verbose per-step trace:
    python -m examples.gemini_episode --target DRD2 --episodes 1 --verbose

Cost (paid tier, per 1M tokens, as of 2025-04):
    gemini-2.5-pro:    $1.25 input / $10.00 output
    gemini-2.5-flash:  $0.30 input / $2.50 output
    gemini-2.0-flash:  $0.10 input / $0.40 output
A 20-turn episode is ~5-15K tokens. 9 eps on 2.5-flash ≈ $0.05-0.15.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from google import genai
from google.genai import types

# ─── .env loading ─────────────────────────────────────────────────────


def _load_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_env()

ENV_URL_DEFAULT = os.environ.get("PHARMARL_ENV_URL", "https://anshumanatrey-pharmarl.hf.space")

PRICING = {
    "gemini-2.5-pro":    {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash":  {"input": 0.30, "output": 2.50},
    "gemini-2.0-flash":  {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash":  {"input": 0.075, "output": 0.30},
}

SYSTEM_PROMPT = """You are a medicinal chemist designing a small-molecule drug. Each turn you edit a molecule by issuing a single JSON action.

Output format (respond with EXACTLY ONE JSON object on its own line — no prose, no markdown fences):
  {"action_type": "ADD_FRAGMENT", "fragment": "<smiles>", "position": 0}
  {"action_type": "REMOVE_FRAGMENT", "position": 0}
  {"action_type": "SUBSTITUTE_ATOM", "position": 0, "new_atom": "F"}
  {"action_type": "TERMINATE"}

Rules:
  - The fragment for ADD_FRAGMENT MUST be in the available_fragments list provided.
  - new_atom for SUBSTITUTE_ATOM is one of C, N, O, F, Cl, Br, I, P, S.
  - TERMINATE is only valid after at least one edit.
  - You are optimizing binding to the target reported in the observation. Higher composite reward = closer to a drug-like high-affinity ligand. Lipinski-passing molecules score better."""


_JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first parseable JSON object from the response."""
    if not text:
        return None
    for m in _JSON_RE.findall(text):
        try:
            return json.loads(m)
        except Exception:
            continue
    return None


def turn_prompt(obs: Dict[str, Any]) -> str:
    p = obs.get("properties", {}) or {}
    return (
        f"Target: {obs.get('target', '?')}\n"
        f"Steps remaining: {obs.get('steps_remaining', '?')}\n"
        f"Current SMILES: {obs.get('smiles', '')}\n"
        f"Properties: MW={p.get('mw', 0):.1f}, LogP={p.get('logp', 0):.2f}, "
        f"HBD={p.get('hbd', 0):.0f}, HBA={p.get('hba', 0):.0f}, "
        f"LipinskiViolations={p.get('lipinski_violations', 0):.0f}\n"
        f"Available fragments: {obs.get('available_fragments', [])}\n"
        f"Valid actions: {obs.get('valid_actions', [])}\n"
        f"Last message: {obs.get('message', '')}\n\n"
        "Respond with one JSON action."
    )


def run_episode(
    client: genai.Client,
    session: requests.Session,
    env_url: str,
    target: str,
    difficulty: str,
    model: str,
    max_steps: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    episode_id = str(uuid.uuid4())
    body = {"difficulty": difficulty, "episode_id": episode_id, "target": target}
    obs = session.post(f"{env_url}/reset", json=body, timeout=30).json()["observation"]

    history: List[Dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    parses_ok = 0
    cumulative = 0.0
    last_smiles = obs["smiles"]

    # Gemini 2.5 thinking budget tuning:
    #   - Flash/Flash-Lite: can disable thinking entirely (thinking_budget=0) — fast & cheap
    #   - Pro: thinking is required; cap at 1024 to avoid consuming the entire output budget
    # max_output_tokens must be > thinking_budget — otherwise the actual response is empty.
    if "flash" in model.lower():
        thinking_budget = 0
        max_out = 200
    else:  # pro / 1.5
        thinking_budget = 1024
        max_out = 2000
    gen_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.4,
        max_output_tokens=max_out,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
    )

    for step_idx in range(max_steps):
        prompt = turn_prompt(obs)
        raw_text = ""
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[prompt],
                config=gen_config,
            )
            raw_text = (resp.text or "").strip()
            if resp.usage_metadata is not None:
                total_input_tokens += resp.usage_metadata.prompt_token_count or 0
                total_output_tokens += resp.usage_metadata.candidates_token_count or 0
        except Exception as e:
            if verbose:
                print(f"    [gemini-err step{step_idx + 1}] {type(e).__name__}: {e}", flush=True)

        action = parse_action(raw_text)
        if action is not None:
            parses_ok += 1
        else:
            # Send a benign no-op so the env charges the parse-fail penalty (-0.5)
            action = {"action_type": "ADD_FRAGMENT", "fragment": "C", "position": 0}

        step_resp = session.post(
            f"{env_url}/step",
            json={"action": action, "episode_id": episode_id},
            timeout=60,
        ).json()
        reward = step_resp["reward"]
        cumulative += reward
        new_obs = step_resp["observation"]

        history.append({
            "step": step_idx + 1,
            "raw": raw_text[:200],
            "action": action,
            "reward": reward,
            "smiles_after": new_obs["smiles"],
        })

        if verbose:
            at = action.get("action_type", "?")
            sm = new_obs["smiles"][:60]
            print(
                f"    step{step_idx + 1:2d}: {at:16s} r={reward:+.3f} "
                f"smiles={sm} done={step_resp['done']}",
                flush=True,
            )

        last_smiles = new_obs["smiles"]
        obs = new_obs
        if step_resp["done"]:
            break

    return {
        "episode_id": episode_id,
        "target": target,
        "difficulty": difficulty,
        "model": model,
        "turns": len(history),
        "cumulative": cumulative,
        "final_smiles": last_smiles,
        "parse_rate": parses_ok / max(1, len(history)),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "tail": history[-3:],
    }


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    p = PRICING.get(model)
    if p is None:
        return 0.0
    return (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--env-url", default=ENV_URL_DEFAULT)
    p.add_argument("--target", nargs="+", default=["DRD2", "GSK3B", "JNK3"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--difficulty", default="easy", choices=("trivial", "easy", "hard"))
    p.add_argument("--model", default="gemini-2.5-flash", choices=list(PRICING.keys()))
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--out", default=None, help="Optional: write per-episode JSON here.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    if not os.environ.get("GEMINI_API_KEY"):
        print("[FAIL] GEMINI_API_KEY missing — copy .env.example to .env and fill in.", file=sys.stderr)
        return 2

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    session = requests.Session()

    try:
        h = session.get(f"{args.env_url}/health", timeout=5).json()
        print(f"[OK]  env reachable: {args.env_url} → {h}")
    except Exception as e:
        print(f"[FAIL] env unreachable at {args.env_url}: {e}", file=sys.stderr)
        return 2

    pricing = PRICING.get(args.model, {})
    print(f"[CFG] model={args.model}  pricing=${pricing.get('input', '?'):.2f}/M in, "
          f"${pricing.get('output', '?'):.2f}/M out")
    print(f"[CFG] targets={args.target}  episodes={args.episodes}  difficulty={args.difficulty}  max_steps={args.max_steps}")
    print()

    all_results: List[Dict[str, Any]] = []
    overall_input = 0
    overall_output = 0
    t0 = time.time()

    for tgt in args.target:
        print(f"=== target = {tgt} ===")
        per_tgt = []
        for ep in range(args.episodes):
            t_ep = time.time()
            r = run_episode(
                client, session, args.env_url, tgt, args.difficulty,
                args.model, args.max_steps, verbose=args.verbose,
            )
            dt = time.time() - t_ep
            per_tgt.append(r)
            all_results.append(r)
            overall_input += r["input_tokens"]
            overall_output += r["output_tokens"]
            print(
                f"  ep{ep + 1}: cum={r['cumulative']:+.3f} "
                f"parse_rate={r['parse_rate'] * 100:.0f}%  "
                f"turns={r['turns']:2d}  "
                f"final={r['final_smiles'][:60]}  ({dt:.1f}s)"
            )
        if per_tgt:
            mean_cum = statistics.mean(r["cumulative"] for r in per_tgt)
            mean_pr = statistics.mean(r["parse_rate"] for r in per_tgt)
            print(f"  → mean cum={mean_cum:+.3f}  mean parse_rate={mean_pr * 100:.0f}%")
        print()

    if not all_results:
        return 0

    grand_mean = statistics.mean(r["cumulative"] for r in all_results)
    grand_pr = statistics.mean(r["parse_rate"] for r in all_results)
    cost = estimate_cost(overall_input, overall_output, args.model)

    print("=" * 60)
    print(f"  OVERALL: {len(all_results)} episodes")
    print(f"  Mean cumulative reward: {grand_mean:+.3f}")
    print(f"  Mean parse_rate: {grand_pr * 100:.0f}%")
    print(f"  Token usage: input={overall_input:,} | output={overall_output:,}")
    print(f"  Est. cost on {args.model}: ${cost:.4f}")
    print(f"  Wall time: {time.time() - t0:.1f}s")
    print("=" * 60)

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"  Per-episode results → {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
