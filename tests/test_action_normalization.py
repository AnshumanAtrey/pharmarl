"""Action normalization — trained LLMs in the wild produce slightly off-schema
JSON ('ACTION' instead of 'action_type', verbose objects with extra keys).
The /step endpoint normalizes these before pydantic validation so the demo
keeps running even when the agent's format discipline slips.

These tests pin the normalizer's behavior so future schema changes don't
silently break LLM-output compatibility.
"""

from __future__ import annotations

from server.app import _normalize_action_dict


def test_canonical_action_type_passes_through_unchanged():
    """The canonical schema must be a no-op."""
    raw = {"action_type": "TERMINATE"}
    out = _normalize_action_dict(raw)
    assert out == raw


def test_uppercase_action_key_is_remapped():
    """Trained LLMs often emit 'ACTION' instead of 'action_type'."""
    out = _normalize_action_dict({"ACTION": "TERMINATE"})
    assert out == {"action_type": "TERMINATE"}


def test_lowercase_action_key_is_remapped():
    out = _normalize_action_dict({"action": "ADD_FRAGMENT", "fragment": "C"})
    assert out["action_type"] == "ADD_FRAGMENT"
    assert out["fragment"] == "C"


def test_verbose_llm_output_strips_noise():
    """A verbose LLM output with SELFIES + FRAGMENTS lists alongside the action
    should be stripped down to just the action schema."""
    raw = {
        "SELFIES": ["[C]", "[O]", "[N]"],
        "FRAGMENTS": ["c1ccccc1", "C(=O)O"],
        "ACTION": "TERMINATE",
        "explanation": "I think we should terminate now",
    }
    out = _normalize_action_dict(raw)
    assert out == {"action_type": "TERMINATE"}


def test_substitute_atom_param_keys_remapped():
    """Variant key names for SUBSTITUTE_ATOM parameters get normalized."""
    out = _normalize_action_dict(
        {"action_type": "SUBSTITUTE_ATOM", "POSITION": 2, "ATOM": "N"}
    )
    assert out == {"action_type": "SUBSTITUTE_ATOM", "position": 2, "new_atom": "N"}


def test_non_dict_input_passed_through():
    """Pydantic should give the clear error, not the normalizer."""
    out = _normalize_action_dict("TERMINATE")
    assert out == "TERMINATE"


def test_action_already_canonical_with_extras_keeps_only_known():
    """If action_type already present but extra noise is there, drop the noise."""
    out = _normalize_action_dict({
        "action_type": "ADD_FRAGMENT",
        "fragment": "C",
        "rationale": "carbon adds methyl group",
    })
    assert out == {"action_type": "ADD_FRAGMENT", "fragment": "C"}
