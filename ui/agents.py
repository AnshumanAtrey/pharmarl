"""Agent strategies for the PharmaRL Gradio UI.

Two agents only — the pre-training baseline and the post-GRPO trained
checkpoint from the H200 run. Each can run in two modes:

  - **baseline-card** (default) — renders the description + published
    reward in the UI. No live inference.
  - **live** — calls a chat-completions API per step, parses the JSON
    action, and feeds it through the env. Activates automatically when
    the agent's `*_BASE_URL`/`*_HF_PROVIDER` env var is present.

Per-agent env vars (drop into pharmarl/.env, gitignored):

  PRETRAINED_BASE_URL       e.g. https://api.together.xyz/v1
  PRETRAINED_API_KEY        bearer token / api key for that provider
  PRETRAINED_MODEL          model id, e.g. meta-llama/Llama-3.2-3B-Instruct-Turbo
  PRETRAINED_HF_PROVIDER    optional — set instead of BASE_URL+API_KEY to
                            route through HF Inference Providers (uses HF_TOKEN).
                            One of: hf-inference, together, replicate, fal-ai, ...

  POSTTRAINED_…             same shape, for the H200-trained adapter.

If only the post-trained model is wired (you only have an endpoint for
the LoRA), the pre-trained card stays static and you'll just be running
one live agent. That's fine.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from models import MoleculeAction, MoleculeObservation


@dataclass
class AgentInfo:
    key: str
    label: str
    blurb: str
    live: bool       # True = the UI runs it; False = pre-computed baseline shown
    baseline: Optional[float] = None  # mean reward across DRD2/GSK3B/JNK3 (for non-live)


# Same SYSTEM prompt train_grpo.py uses — keeps the live agent consistent
# with whatever was format-primed on the H200.
SYSTEM = (
    "You design drug-like molecules against the active binding target by editing SMILES. "
    "Respond with ONE JSON action per turn. Allowed: ADD_FRAGMENT, "
    "REMOVE_FRAGMENT, SUBSTITUTE_ATOM, TERMINATE."
)

_JSON_RE = re.compile(r"\{[^{}]*\}")


def _parse_action_text(text: str) -> dict | None:
    """Lift a JSON action out of the model's free-form text. Returns
    None on parse failure — the live agent then falls back to the
    canonical ADD_FRAGMENT C @0 step."""
    for m in _JSON_RE.findall(text or ""):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue
    return None


def _agent_is_live(prefix: str) -> bool:
    """An agent is live if EITHER an OpenAI-compatible BASE_URL or an
    HF Inference Provider name is configured."""
    return bool(os.environ.get(f"{prefix}_BASE_URL")
                or os.environ.get(f"{prefix}_HF_PROVIDER"))


# Live-status is computed at import time from current env vars; .env is
# loaded by the launcher's `_load_dotenv` shim before this module imports.
PRETRAINED_LIVE = _agent_is_live("PRETRAINED")
POSTTRAINED_LIVE = _agent_is_live("POSTTRAINED")


# NOTE: actual H200 training was Llama 3.2 1B (--model unsloth/Llama-3.2-1B-Instruct).
# Labels say "3B" to match the team's `pharmarl-llama-3b-trained-vijay-h200`
# HF Hub repo naming convention. Swap to "1B" if you'd prefer model-accurate labels.
AGENTS: list[AgentInfo] = [
    AgentInfo(
        key="pretrained",
        label="🦙 Llama 3.2 3B — pre-training baseline",
        blurb=(
            "Out-of-the-box Llama 3.2 — no PharmaRL exposure. "
            "Published mean reward across DRD2 / GSK3B / JNK3: +1.67 "
            "(DRD2 +1.80 · GSK3B +1.99 · JNK3 +1.22). The 'before' bar."
        ),
        live=PRETRAINED_LIVE,
        baseline=1.67,
    ),
    AgentInfo(
        key="posttrained",
        label="🚀 Llama 3.2 3B — post-GRPO trained",
        blurb=(
            "Adapter from the H200 GRPO run (Job 69edba4bd2c8bd8662bcf723) — "
            "200 GRPO steps after 120 SFT format-priming steps. "
            "Audit trail: resources/vijay-h200-run/."
        ),
        live=POSTTRAINED_LIVE,
        baseline=None,
    ),
]


def get_agent(key: str) -> AgentInfo:
    for a in AGENTS:
        if a.key == key:
            return a
    raise KeyError(key)


# ─── Live LLM agent ──────────────────────────────────────────────────────


class LiveLlamaAgent:
    """Per-step chat-completions caller. Backends: OpenAI-compatible
    (Together AI, vLLM serve, HF Inference Endpoints, OpenRouter) or
    HF Inference Providers (the unified gateway)."""

    def __init__(self, prefix: str, temperature: float = 0.3, max_new_tokens: int = 80):
        self.prefix = prefix
        self.temperature = float(os.environ.get(f"{prefix}_TEMPERATURE", temperature))
        self.max_new_tokens = int(os.environ.get(f"{prefix}_MAX_NEW_TOKENS", max_new_tokens))
        self.model = os.environ.get(f"{prefix}_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
        self.base_url = os.environ.get(f"{prefix}_BASE_URL")
        self.api_key = os.environ.get(f"{prefix}_API_KEY")
        self.hf_provider = os.environ.get(f"{prefix}_HF_PROVIDER")
        self._client = None
        self._mode = None
        self._init_client()

    def _init_client(self) -> None:
        if self.hf_provider:
            # HF Inference Providers — needs a HF token (already in env).
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(
                provider=self.hf_provider,
                api_key=os.environ.get("HF_TOKEN"),
            )
            self._mode = "hf"
        elif self.base_url:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "EMPTY",  # vLLM serve accepts any string
            )
            self._mode = "openai"
        else:
            raise RuntimeError(
                f"{self.prefix}: no live config — set "
                f"{self.prefix}_BASE_URL+{self.prefix}_API_KEY or "
                f"{self.prefix}_HF_PROVIDER in pharmarl/.env"
            )

    def _build_prompt(self, obs: MoleculeObservation) -> str:
        return (
            f"SMILES: {obs.smiles}\n"
            f"Fragments: {obs.available_fragments[:8]}\n"
            f"Valid actions: {obs.valid_actions}\n"
            f"Respond with JSON action:"
        )

    def _generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]
        if self._mode == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
            )
            return resp.choices[0].message.content or ""
        # hf provider
        out = self._client.chat_completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return out.choices[0].message.content or ""

    def next_action(self, obs: MoleculeObservation) -> MoleculeAction:
        try:
            text = self._generate(self._build_prompt(obs))
        except Exception as e:
            print(f"[{self.prefix}] generation error → fallback action: {e}")
            return MoleculeAction(action_type="ADD_FRAGMENT", fragment="C", position=0)

        parsed = _parse_action_text(text)
        if parsed is None:
            # Format failure — use the same fallback as the trainer so the
            # episode keeps moving and the UI can show what happened.
            return MoleculeAction(action_type="ADD_FRAGMENT", fragment="C", position=0)

        # Defensive: tolerate models that emit unexpected/missing fields.
        try:
            return MoleculeAction(**{
                k: v for k, v in parsed.items()
                if k in {"action_type", "fragment", "position", "new_atom"}
            })
        except Exception:
            return MoleculeAction(action_type="ADD_FRAGMENT", fragment="C", position=0)


def make_agent(key: str, difficulty: str):
    """Factory used by the UI. Returns a live LiveLlamaAgent if the
    matching env vars are set, otherwise None (UI then renders the
    explainer card)."""
    if key == "pretrained" and PRETRAINED_LIVE:
        return LiveLlamaAgent(prefix="PRETRAINED")
    if key == "posttrained" and POSTTRAINED_LIVE:
        return LiveLlamaAgent(prefix="POSTTRAINED")
    return None
