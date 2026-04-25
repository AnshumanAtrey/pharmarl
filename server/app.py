"""FastAPI app for PharmaRL — wraps DrugDiscoveryEnvironment in OpenEnv server.

Endpoints:
  POST /reset        → start (or restart) episode; returns observation incl. episode_id
  POST /step         → apply action to a given episode (episode_id required for multi-step)
  GET  /state        → inspect current state
  GET  /health       → liveness check

NOTE on session handling
------------------------
OpenEnv's stock /reset and /step handlers create a NEW Environment instance per
HTTP request and discard it afterward (see http_server.py:582,617). That works
for stateless graders (Type A) but breaks multi-step state-dynamics envs like
ours: every /step would see step_count=0, so the episode could never reach
TERMINATE or max_steps.

We override /reset and /step to maintain a session dict keyed by `episode_id`.
The agent sends `episode_id` (any unique string per concurrent rollout) on each
call. /reset creates or reuses the env for that id; /step looks it up.

This unblocks GRPO with G=8 parallel rollouts: callers issue 8 different
`episode_id`s and the server keeps 8 envs alive concurrently.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Optional
from uuid import uuid4

from fastapi import Body, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.serialization import serialize_observation
except ImportError as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

try:
    from models import MoleculeAction, MoleculeObservation
    from server.drug_discovery_environment import DrugDiscoveryEnvironment
except ImportError:
    from ..models import MoleculeAction, MoleculeObservation
    from .drug_discovery_environment import DrugDiscoveryEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pharmarl")


# Build the base app with OpenEnv's machinery (gives us /metadata, /schema, /ws, /mcp).
app = create_app(
    DrugDiscoveryEnvironment,
    MoleculeAction,
    MoleculeObservation,
    env_name="pharmarl",
    max_concurrent_envs=64,
)


# ─── Session-keyed env registry ────────────────────────────────────────
_envs: Dict[str, DrugDiscoveryEnvironment] = {}
_envs_lock = threading.Lock()


class _ResetBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    episode_id: Optional[str] = Field(default=None, max_length=255)
    seed: Optional[int] = Field(default=None, ge=0)
    difficulty: Optional[str] = None
    training_step: Optional[int] = None


class _StepBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    action: Dict
    episode_id: Optional[str] = Field(default=None, max_length=255)


def _get_or_create_env(episode_id: str) -> DrugDiscoveryEnvironment:
    with _envs_lock:
        env = _envs.get(episode_id)
        if env is None:
            env = DrugDiscoveryEnvironment()
            _envs[episode_id] = env
        return env


def _drop_env(episode_id: str) -> None:
    with _envs_lock:
        env = _envs.pop(episode_id, None)
    if env is not None:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            logger.warning("env.close() failed for %s", episode_id, exc_info=True)


# Strip the stateless handlers OpenEnv registered, then re-register stateful ones.
app.router.routes = [
    r for r in app.router.routes
    if not (getattr(r, "path", None) in {"/reset", "/step"})
]


@app.post("/reset")
async def reset_endpoint(body: _ResetBody = Body(default_factory=_ResetBody)):
    episode_id = body.episode_id or str(uuid4())
    env = _get_or_create_env(episode_id)

    kwargs = {}
    if body.seed is not None:
        kwargs["seed"] = body.seed
    if body.difficulty is not None:
        kwargs["difficulty"] = body.difficulty
    if body.training_step is not None:
        kwargs["training_step"] = body.training_step
    kwargs["episode_id"] = episode_id

    observation = env.reset(**kwargs)
    payload = serialize_observation(observation)
    payload["observation"]["episode_id"] = episode_id
    return JSONResponse(payload)


@app.post("/step")
async def step_endpoint(body: _StepBody):
    if not body.episode_id:
        raise HTTPException(
            status_code=400,
            detail="episode_id is required on /step. Get one from /reset.",
        )

    with _envs_lock:
        env = _envs.get(body.episode_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown episode_id={body.episode_id!r}; call /reset first.",
        )

    try:
        action = MoleculeAction(**body.action)
    except Exception as e:  # pydantic ValidationError
        raise HTTPException(status_code=422, detail=str(e))

    observation = env.step(action)
    payload = serialize_observation(observation)
    payload["observation"]["episode_id"] = body.episode_id

    if observation.done:
        _drop_env(body.episode_id)

    return JSONResponse(payload)


# ─── Health check ─────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "environment": "pharmarl"}


# ─── Active sessions (debug + capacity) ──────────────────────────────
@app.get("/sessions")
async def sessions():
    with _envs_lock:
        return {"active_episodes": list(_envs.keys()), "count": len(_envs)}


# ─── Tasks endpoint (hackathon convention from Round 1) ──────────────
@app.get("/tasks")
async def get_tasks():
    """Lists curriculum tiers + action schema + active target."""
    from server.curriculum import DEFAULT_CONFIG
    from server.oracles import get_active_target_name

    return JSONResponse(
        {
            "tasks": [
                {
                    "id": "trivial",
                    "max_steps": DEFAULT_CONFIG.trivial_max_steps,
                    "components": list(DEFAULT_CONFIG.trivial_components),
                },
                {
                    "id": "easy",
                    "max_steps": DEFAULT_CONFIG.easy_max_steps,
                    "components": list(DEFAULT_CONFIG.easy_components),
                },
                {
                    "id": "hard",
                    "max_steps": DEFAULT_CONFIG.hard_max_steps,
                    "components": list(DEFAULT_CONFIG.hard_components),
                },
            ],
            "action_schema": MoleculeAction.model_json_schema(),
            "target": get_active_target_name(),
        }
    )


# ─── Oracle diagnostics endpoint ─────────────────────────────────────
@app.get("/oracle_status")
async def oracle_status():
    """Reports the live binding oracle + the human-readable target it scores."""
    from server.oracles import get_active_oracle_name, get_active_target_name

    return {
        "active_oracle": get_active_oracle_name(),
        "active_target": get_active_target_name(),
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point: `python -m server.app`."""
    import uvicorn
    port = int(os.environ.get("PORT", port))
    host = os.environ.get("HOST", host)
    logger.info("Starting PharmaRL on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
