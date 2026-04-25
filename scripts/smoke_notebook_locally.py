"""Local smoke test of the Colab notebook's HTTP loop — no LLM, no GPU.

Reproduces:
  - notebook cell 8  (/reset shape)
  - notebook cell 10 (parse_action, reward_format, reward_action_valid, reward_env)
  - notebook cell 12 (rollout_episode, with a random-action stub instead of the LLM)
  - notebook cell 14 (5-episode baseline run + cumulative reward print)

Run a local server first:
    uvicorn server.app:app --port 8000

Then:
    python scripts/smoke_notebook_locally.py
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
from typing import Any, Dict, List, Optional

import requests

ENV_URL = "http://127.0.0.1:8000"
SESSION = requests.Session()  # connection pool — avoid Mac EADDRNOTAVAIL on tight loops

# ---- Lifted verbatim from notebook cell 10 ----
_JSON_RE = re.compile(r"\{[^{}]*\}")


def parse_action(text: str) -> Optional[Dict[str, Any]]:
    for m in _JSON_RE.findall(text):
        try:
            return json.loads(m)
        except Exception:
            continue
    return None


def reward_format(prompts, completions, **kwargs):
    return [0.5 if parse_action(c) is not None else -0.5 for c in completions]


def reward_action_valid(prompts, completions, env_responses=None, **kwargs):
    out = []
    for env_r in (env_responses or [None] * len(completions)):
        if env_r is None:
            out.append(0.0)
            continue
        out.append(0.1 if env_r["observation"].get("last_action_valid") else -0.1)
    return out


def reward_env(prompts, completions, env_responses=None, **kwargs):
    out = []
    for env_r in (env_responses or [None] * len(completions)):
        out.append(float(env_r["reward"]) if env_r else 0.0)
    return out


# ---- Stub rollout — replaces the LLM-driven rollout from notebook cell 12 ----


def stub_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Pick a random valid action from the env's reported `valid_actions`."""
    action_type = random.choice(obs["valid_actions"])
    if action_type == "ADD_FRAGMENT":
        return {
            "action_type": "ADD_FRAGMENT",
            "fragment": random.choice(obs["available_fragments"]),
            "position": 0,
        }
    if action_type == "REMOVE_FRAGMENT":
        return {"action_type": "REMOVE_FRAGMENT", "position": 0}
    if action_type == "SUBSTITUTE_ATOM":
        return {
            "action_type": "SUBSTITUTE_ATOM",
            "position": 0,
            "new_atom": random.choice(["C", "N", "O", "F"]),
        }
    return {"action_type": "TERMINATE"}


def rollout_episode(env_url: str, difficulty: str = "trivial", verbose: bool = False, max_safety: int = 50):
    """Mirrors notebook cell 12 — uses episode_id to keep state across HTTP calls.

    OpenEnv's stock /reset+/step are stateless (each request creates a fresh env).
    We pass episode_id on every call so the server can route to the persistent
    env for this rollout.
    """
    import time
    import uuid
    episode_id = str(uuid.uuid4())
    r = SESSION.post(
        f"{env_url}/reset",
        json={"difficulty": difficulty, "episode_id": episode_id},
    ).json()
    obs = r["observation"]
    actions, rewards, env_responses = [], [], []
    cumulative = 0.0
    step_no = 0
    while True:
        step_no += 1
        action = stub_action(obs)
        t0 = time.time()
        step = SESSION.post(
            f"{env_url}/step",
            json={"action": action, "episode_id": episode_id},
            timeout=60,
        ).json()
        dt = time.time() - t0
        if verbose:
            print(f"    step{step_no:2d}: {action['action_type']:18s} r={step['reward']:+.3f} done={step['done']} trunc={step['observation'].get('truncated')} ({dt:.2f}s)", flush=True)
        actions.append(action)
        rewards.append(step["reward"])
        env_responses.append(step)
        cumulative += step["reward"]
        obs = step["observation"]
        if step["done"]:
            break
        if step_no >= max_safety:
            print(f"    [SAFETY-BREAK] step_no={step_no} hit max_safety={max_safety} without done", flush=True)
            break
    return {
        "actions": actions,
        "rewards": rewards,
        "env_responses": env_responses,
        "cumulative": cumulative,
        "final_smiles": obs["smiles"],
    }


def main() -> int:
    random.seed(42)

    print("=" * 60)
    print("Local notebook smoke test")
    print("=" * 60)

    # Server reachable?
    try:
        h = SESSION.get(f"{ENV_URL}/health", timeout=2).json()
        assert h.get("status") == "healthy", h
        print(f"[OK] /health: {h}")
    except Exception as e:
        print(f"[FAIL] server unreachable at {ENV_URL}: {e}")
        return 1

    # /reset shape matches notebook cell 8 expectations
    r = SESSION.post(f"{ENV_URL}/reset", json={"difficulty": "trivial"}).json()
    obs = r["observation"]
    required_keys = {
        "smiles", "selfies", "target", "difficulty", "properties",
        "valid_actions", "available_fragments", "steps_remaining",
        "last_action_valid", "message", "truncated",
    }
    missing = required_keys - obs.keys()
    if missing:
        print(f"[FAIL] /reset missing keys: {missing}")
        return 1
    print(f"[OK]  /reset returns all expected keys: smiles={obs['smiles']}")

    # /step shape matches notebook cell 12 expectations
    # First /reset to register an episode, then /step against it
    import uuid
    probe_eid = str(uuid.uuid4())
    SESSION.post(f"{ENV_URL}/reset", json={"difficulty": "trivial", "episode_id": probe_eid}).json()
    s = SESSION.post(
        f"{ENV_URL}/step",
        json={
            "action": {"action_type": "ADD_FRAGMENT", "fragment": "C", "position": 0},
            "episode_id": probe_eid,
        },
    ).json()
    if "reward" not in s or "observation" not in s or "done" not in s:
        print(f"[FAIL] /step shape: {s}")
        return 1
    print(f"[OK]  /step returns reward={s['reward']} done={s['done']}")

    # parse_action regex
    samples = [
        ('Sure! {"action_type": "TERMINATE"}', True),
        ('{"action_type": "ADD_FRAGMENT", "fragment": "C", "position": 0}', True),
        ("no JSON here", False),
        ('Action: {"action_type": "ADD_FRAGMENT", "fragment": "c1ccccc1"} cool', True),
    ]
    for text, should_parse in samples:
        parsed = parse_action(text)
        ok = (parsed is not None) == should_parse
        print(f"[{'OK' if ok else 'FAIL'}]   parse_action({text[:40]!r:42s}) → {parsed}")
        if not ok:
            return 1

    # reward_format / reward_action_valid / reward_env
    completions = [
        '{"action_type": "TERMINATE"}',
        "garbage no json",
    ]
    rf = reward_format(None, completions)
    assert rf == [0.5, -0.5], rf
    print(f"[OK]  reward_format: {rf}")

    mock_env_resp = [
        {"observation": {"last_action_valid": True}, "reward": 0.05},
        {"observation": {"last_action_valid": False}, "reward": -0.5},
    ]
    rav = reward_action_valid(None, completions, env_responses=mock_env_resp)
    assert rav == [0.1, -0.1], rav
    print(f"[OK]  reward_action_valid: {rav}")

    re_ = reward_env(None, completions, env_responses=mock_env_resp)
    assert re_ == [0.05, -0.5], re_
    print(f"[OK]  reward_env: {re_}")

    # Full rollout — 5 episodes
    print("\n--- 5-episode random-action baseline ---", flush=True)
    import time
    eps = []
    for i in range(5):
        t0 = time.time()
        ep = rollout_episode(ENV_URL, "trivial", verbose=(i == 0))
        eps.append(ep)
        print(
            f"  ep{i+1}: steps={len(ep['actions'])} cum={ep['cumulative']:+.3f} "
            f"final={ep['final_smiles']} ({time.time()-t0:.1f}s)",
            flush=True,
        )
    avg = statistics.mean(e["cumulative"] for e in eps)
    print(f"\n  avg cumulative reward: {avg:+.3f}")

    # Sanity: cumulative reward should not be NaN; at least one episode terminated
    if any(e["cumulative"] != e["cumulative"] for e in eps):  # NaN check
        print("[FAIL] NaN cumulative reward")
        return 1

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    print("\nKnown notebook issue to fix before Colab run:")
    print("  cell 12: requests.post(f'{env_url}/step', json=action)")
    print("  must be: requests.post(f'{env_url}/step', json={'action': action})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
