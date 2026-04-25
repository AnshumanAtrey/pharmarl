"""Tests for the Fleet AI oversight LLM module.

All tests are HERMETIC — no real API calls. The LLM call layer is patched
via monkey-patch so we can validate the prompt construction, response
parsing, error handling, and env integration without burning quota.
"""

from __future__ import annotations

import json

import pytest

from server.oversight import (
    LLMOversight,
    OversightReport,
    _build_user_prompt,
    _parse_oversight_response,
    reset_default_oversight,
)


# ─── Response parsing ─────────────────────────────────────────────────


def test_parse_clean_json_response() -> None:
    raw = """
    Some preamble. Here is the analysis:
    {"strategy_summary": "Agent added benzene then terminated",
     "risk_flags": ["short_episode"],
     "risk_level": "low",
     "explanation": "Behaved reasonably"}
    """
    report = _parse_oversight_response(raw, model_name="test-model")
    assert report.strategy_summary == "Agent added benzene then terminated"
    assert report.risk_flags == ["short_episode"]
    assert report.risk_level == "low"
    assert report.explanation == "Behaved reasonably"
    assert report.model_name == "test-model"


def test_parse_response_without_json_returns_unknown() -> None:
    raw = "I cannot do that, Dave."
    report = _parse_oversight_response(raw)
    assert report.risk_level == "unknown"
    assert "parse_failed_no_json" in report.risk_flags


def test_parse_response_with_invalid_json_returns_unknown() -> None:
    raw = "{this is not valid json"
    report = _parse_oversight_response(raw)
    assert report.risk_level == "unknown"
    assert any("parse_failed" in f for f in report.risk_flags)


def test_parse_response_clamps_invalid_risk_level() -> None:
    """If LLM hallucinates a risk_level not in the allowed set, clamp to 'unknown'."""
    raw = '{"strategy_summary": "x", "risk_flags": [], "risk_level": "catastrophic", "explanation": "y"}'
    report = _parse_oversight_response(raw)
    assert report.risk_level == "unknown"


def test_parse_response_coerces_non_list_flags() -> None:
    """Flags should always end up as a list of strings, even if LLM returned a string."""
    raw = '{"strategy_summary": "x", "risk_flags": "single_flag", "risk_level": "low", "explanation": "y"}'
    report = _parse_oversight_response(raw)
    assert report.risk_flags == ["single_flag"]


# ─── Availability gating ─────────────────────────────────────────────


def test_is_available_false_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reset_default_oversight()
    o = LLMOversight(provider="openrouter", api_key=None)
    assert o.is_available() is False


def test_is_available_true_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
    reset_default_oversight()
    o = LLMOversight(provider="openrouter")
    assert o.is_available() is True


def test_analyze_without_key_returns_unknown_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    o = LLMOversight(provider="openrouter", api_key=None)
    report = o.analyze(
        target="DRD2",
        starting_smiles="C",
        final_smiles="CC",
        action_history=[{"action": "ADD_FRAGMENT", "before": "C", "after": "CC"}],
        final_reward=2.5,
        lipinski_passes=True,
    )
    assert report.risk_level == "unknown"
    assert "oversight_unavailable" in report.risk_flags


# ─── LLM error handling ─────────────────────────────────────────────


def test_analyze_handles_request_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the HTTP layer raises, oversight returns 'unknown', not crashes."""
    o = LLMOversight(provider="openrouter", api_key="sk-test-fake")

    def boom(self, prompt: str) -> str:
        raise ConnectionError("simulated network failure")

    monkeypatch.setattr(LLMOversight, "_call_llm", boom)
    report = o.analyze(
        target="DRD2",
        starting_smiles="C",
        final_smiles="CC",
        action_history=[{"action": "ADD_FRAGMENT", "before": "C", "after": "CC"}],
        final_reward=2.5,
        lipinski_passes=True,
    )
    assert report.risk_level == "unknown"
    assert "llm_call_failed" in report.risk_flags


def test_analyze_with_mocked_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path — LLM returns valid JSON, parsed correctly."""
    o = LLMOversight(provider="openrouter", api_key="sk-test-fake")

    fake_response = json.dumps({
        "strategy_summary": "Built a benzene-amine scaffold",
        "risk_flags": ["short_episode"],
        "risk_level": "low",
        "explanation": "Reasonable trajectory",
    })

    def mocked(self, prompt: str) -> str:
        return fake_response

    monkeypatch.setattr(LLMOversight, "_call_llm", mocked)
    report = o.analyze(
        target="DRD2",
        starting_smiles="C",
        final_smiles="c1ccncc1",
        action_history=[
            {"action": "ADD_FRAGMENT", "before": "C", "after": "Cc1ccccc1"},
            {"action": "TERMINATE", "before": "Cc1ccccc1", "after": "c1ccncc1"},
        ],
        final_reward=4.5,
        lipinski_passes=True,
    )
    assert report.risk_level == "low"
    assert report.strategy_summary == "Built a benzene-amine scaffold"
    assert "short_episode" in report.risk_flags


# ─── Prompt construction ────────────────────────────────────────────


def test_user_prompt_includes_target_and_history() -> None:
    prompt = _build_user_prompt(
        target="GSK3B",
        starting_smiles="C",
        final_smiles="Cc1ccccc1",
        action_history=[
            {"action": "ADD_FRAGMENT", "before": "C", "after": "Cc1ccccc1"},
        ],
        final_reward=3.2,
        lipinski_passes=True,
    )
    assert "GSK3B" in prompt
    assert "Cc1ccccc1" in prompt
    assert "PASS" in prompt
    assert "3.2" in prompt or "3.20" in prompt or "3.200" in prompt


def test_user_prompt_caps_history_at_20_steps() -> None:
    """For prompt-size sanity, action history is truncated."""
    long_history = [
        {"action": "ADD_FRAGMENT", "before": "C", "after": f"step_{i}"}
        for i in range(50)
    ]
    prompt = _build_user_prompt(
        target="DRD2",
        starting_smiles="C",
        final_smiles="X",
        action_history=long_history,
        final_reward=5.0,
        lipinski_passes=False,
    )
    # step_0 through step_19 should be in the prompt; step_25 should NOT.
    assert "step_0" in prompt
    assert "step_19" in prompt
    assert "step_25" not in prompt


# ─── Env integration ────────────────────────────────────────────────


def test_env_oversight_disabled_by_default() -> None:
    """Default config has oversight_enabled=False — no oversight key in metadata."""
    from server.curriculum import CurriculumConfig
    from server.drug_discovery_environment import DrugDiscoveryEnvironment

    config = CurriculumConfig()
    assert config.oversight_enabled is False

    env = DrugDiscoveryEnvironment(seed=0, config=config)
    obs = env.reset(difficulty="trivial")
    assert "oversight" not in obs.metadata


def test_env_oversight_only_runs_at_done(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even with oversight_enabled=True, no oversight call until episode ends."""
    from dataclasses import replace

    from server.curriculum import DEFAULT_CONFIG
    from server.drug_discovery_environment import DrugDiscoveryEnvironment

    cfg = replace(DEFAULT_CONFIG, oversight_enabled=True)
    env = DrugDiscoveryEnvironment(seed=0, config=cfg)

    # Patch LLM call so we can verify it's called the right number of times
    call_count = {"n": 0}

    def fake_call(self, prompt: str) -> str:
        call_count["n"] += 1
        return json.dumps({
            "strategy_summary": "test",
            "risk_flags": [],
            "risk_level": "low",
            "explanation": "ok",
        })

    monkeypatch.setattr(LLMOversight, "_call_llm", fake_call)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    obs = env.reset(difficulty="trivial")
    # Take a non-TERMINATE action — oversight must NOT fire
    from models import MoleculeAction
    obs = env.step(MoleculeAction(action_type="ADD_FRAGMENT", fragment="C", position=0))
    assert call_count["n"] == 0
    assert "oversight" not in obs.metadata
