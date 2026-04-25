"""FastAPI app for PharmaRL — wraps DrugDiscoveryEnvironment in OpenEnv server.

Endpoints (provided by openenv.core.env_server.http_server.create_app):
  POST /reset        → start episode
  POST /step         → apply action
  GET  /state        → inspect current state
  GET  /health       → liveness check (HF Space requirement)
"""

from __future__ import annotations

import logging
import os

try:
    from openenv.core.env_server.http_server import create_app
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

from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pharmarl")


app = create_app(
    DrugDiscoveryEnvironment,
    MoleculeAction,
    MoleculeObservation,
    env_name="pharmarl",
    max_concurrent_envs=4,
)


# ─── Health check ─────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "healthy", "environment": "pharmarl"}


# ─── Tasks endpoint (hackathon convention from Round 1) ──────────────
@app.get("/tasks")
async def get_tasks():
    """Lists curriculum tiers + action schema."""
    from server.curriculum import DEFAULT_CONFIG

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
            "target": "SARS-CoV-2_Mpro",
        }
    )


# ─── Oracle diagnostics endpoint ─────────────────────────────────────
@app.get("/oracle_status")
async def oracle_status():
    """Reports which docking oracle is active (Mpro vs DRD2 fallback)."""
    from server.oracles.docking_mpro import get_active_oracle_name

    return {"active_docking_oracle": get_active_oracle_name()}


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point: `python -m server.app`."""
    import uvicorn
    port = int(os.environ.get("PORT", port))
    host = os.environ.get("HOST", host)
    logger.info("Starting PharmaRL on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
