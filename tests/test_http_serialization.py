"""Pinning tests for issues #8 and #9 — make sure the HTTP layer surfaces
critique + oversight as top-level fields, and that /reset accepts the
opt-in mechanic flags.

If these regress, the deployed env will silently drop sub-theme features
when external clients (trained Qwen on HF Space, OpenRouter / Gemini agents)
call /reset and /step over HTTP.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ─── Issue #8 — top-level critique + oversight in HTTP response ──────


def test_critique_is_top_level_field_in_http_response() -> None:
    """When critic_enabled, observation should include 'critique' at top level
    (not buried in metadata that OpenEnv strips during serialization)."""
    from server.app import app

    client = TestClient(app)
    r = client.post("/reset", json={"difficulty": "easy", "critic_enabled": True})
    assert r.status_code == 200
    obs = r.json()["observation"]
    assert "critique" in obs, f"missing 'critique'; got keys: {sorted(obs)}"
    # Allow either dict (when populated) or null (e.g., empty molecule edge case),
    # but the FIELD must exist on the wire.
    assert obs["critique"] is None or isinstance(obs["critique"], dict)


def test_oversight_is_top_level_field_in_http_response() -> None:
    """Field must exist as a top-level key even when oversight is disabled
    (default None) so external clients can safely access obs['oversight']."""
    from server.app import app

    client = TestClient(app)
    r = client.post("/reset", json={"difficulty": "easy"})
    assert r.status_code == 200
    obs = r.json()["observation"]
    assert "oversight" in obs, f"missing 'oversight'; got keys: {sorted(obs)}"
    assert obs["oversight"] is None  # default OFF


# ─── Issue #9 — /reset accepts mechanic flags as per-episode overrides ──


def test_reset_accepts_critic_enabled_flag() -> None:
    """/reset with critic_enabled=true should populate critique."""
    from server.app import app

    client = TestClient(app)
    r = client.post("/reset", json={"difficulty": "easy", "critic_enabled": True})
    assert r.status_code == 200
    obs = r.json()["observation"]
    # The reset itself happens with last_action_valid=True and a non-empty smiles,
    # so the critic should populate critique.
    assert obs.get("critique") is not None, "critic_enabled=true should populate critique"


def test_reset_accepts_schema_drift_enabled_flag() -> None:
    """/reset with schema_drift_enabled=true + drift_profile=early_admet should
    apply that profile's weights — observable through active_constraints
    (early_admet pre-drift weights are docking+qed only, no SA / no toxicity)."""
    from server.app import app

    client = TestClient(app)
    r = client.post(
        "/reset",
        json={
            "difficulty": "easy",
            "schema_drift_enabled": True,
            "drift_profile": "early_admet",
        },
    )
    assert r.status_code == 200
    obs = r.json()["observation"]
    active = obs.get("active_constraints", [])
    # early_admet pre-drift = (0.60 docking, 0.40 qed, 0.00 sa, 0.00 tox)
    # → active_constraints should be ['docking', 'qed'] only
    assert "docking" in active and "qed" in active, (
        f"early_admet should activate docking+qed; got {active}"
    )
    assert "sa" not in active and "toxicity" not in active, (
        f"early_admet pre-drift should NOT activate sa/toxicity; got {active}"
    )


def test_reset_accepts_oversight_enabled_flag_without_calling_llm() -> None:
    """/reset with oversight_enabled=true should not call the LLM (oversight
    only fires at TERMINATE). The reset response should still 200."""
    from server.app import app

    client = TestClient(app)
    r = client.post("/reset", json={"difficulty": "easy", "oversight_enabled": True})
    assert r.status_code == 200
    obs = r.json()["observation"]
    # Reset is not an end-of-episode boundary, so oversight should be None
    assert obs.get("oversight") is None


def test_reset_with_no_flags_keeps_defaults_off() -> None:
    """Backwards compat — /reset without any flags should leave critic/oversight OFF."""
    from server.app import app

    client = TestClient(app)
    r = client.post("/reset", json={"difficulty": "easy"})
    assert r.status_code == 200
    obs = r.json()["observation"]
    assert obs.get("critique") is None
    assert obs.get("oversight") is None


def test_reset_unknown_extra_field_is_ignored() -> None:
    """Future-proofing — /reset with an unknown extra field should not 422.
    The pydantic config has extra='allow'."""
    from server.app import app

    client = TestClient(app)
    r = client.post(
        "/reset",
        json={"difficulty": "easy", "future_unknown_flag": True},
    )
    assert r.status_code == 200
