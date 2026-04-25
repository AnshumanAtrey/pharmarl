"""Tests for the rules-based medicinal-chemist critic (Halluminate sub-theme).

The critic is a deterministic substructure / property reviewer that runs after
the policy proposes an edit. These tests cover:

  - invalid SMILES → reject verdict
  - high-MW polyaromatic → MW_TOO_HIGH warning
  - drug-like aspirin → approve (no warnings, no errors)
  - thiocarbonyl-containing fragment → PAINS warning
  - reproducibility / determinism (same input → same critique)
  - critic disabled by default (env behavior unchanged)
  - critic enabled → observation.metadata.critique populated
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MoleculeAction
from server.critic import Critique, MedChemCritic, default_critic
from server.curriculum import CurriculumConfig
from server.drug_discovery_environment import DrugDiscoveryEnvironment


# ─────────────────────────────────────────────────────────────────────────────
# Critic unit tests
# ─────────────────────────────────────────────────────────────────────────────


def test_critic_rejects_invalid_smiles() -> None:
    """Garbled SMILES that won't parse should yield a 'reject' verdict."""
    c = default_critic.critique("@@@invalid@@@")
    assert c.overall == "reject"
    assert any(i.severity == "error" and i.code == "INVALID" for i in c.issues)


def test_critic_warns_about_high_mw() -> None:
    """A large multi-amide blob (MW > 700) should trigger the MW_TOO_HIGH warning."""
    big = (
        "O=C(NCc1ccc(NC(=O)Nc2ccc(NC(=O)c3ccccc3)cc2)cc1)"
        "c1ccc(NC(=O)Nc2ccc(NC(=O)c3ccccc3)cc2)cc1"
    )
    c = default_critic.critique(big)
    codes = {i.code for i in c.issues}
    assert "MW_TOO_HIGH" in codes


def test_critic_approves_drug_like_molecule() -> None:
    """Aspirin (CC(=O)Oc1ccccc1C(=O)O) is the canonical drug-like molecule.

    No warnings, no errors → 'approve'.
    """
    c = default_critic.critique("CC(=O)Oc1ccccc1C(=O)O")
    assert c.overall == "approve"
    assert all(i.severity != "warning" for i in c.issues)


def test_critic_flags_pains_pattern() -> None:
    """A thiocarbonyl-containing molecule triggers the PAINS_THIOCARBONYL warning."""
    c = default_critic.critique("CC(=S)NC")
    codes = {i.code for i in c.issues}
    assert any(code.startswith("PAINS_") for code in codes), (
        f"expected at least one PAINS_* code, got {codes}"
    )


def test_critic_is_deterministic() -> None:
    """Same SMILES → same critique. Crucial for reproducible critic-conditioned training."""
    smi = "CC(=O)Oc1ccccc1C(=O)O"
    a = default_critic.critique(smi)
    b = default_critic.critique(smi)
    assert a.overall == b.overall
    assert [(i.code, i.severity) for i in a.issues] == [(i.code, i.severity) for i in b.issues]


def test_critic_too_small_flag() -> None:
    """Methane has too few heavy atoms — surface a TOO_SMALL info issue."""
    c = default_critic.critique("C")
    codes = {i.code for i in c.issues}
    assert "TOO_SMALL" in codes


# ─────────────────────────────────────────────────────────────────────────────
# Env-integration tests — critic is gated behind config.critic_enabled
# ─────────────────────────────────────────────────────────────────────────────


def test_critic_disabled_by_default() -> None:
    """Without critic_enabled flag, observation.metadata has no 'critique' key.

    Default OFF — the headline training run is unaffected.
    """
    env = DrugDiscoveryEnvironment(seed=42)  # default config → critic_enabled=False
    env.reset(difficulty="trivial")
    obs = env.step(MoleculeAction(action_type="ADD_FRAGMENT", fragment="C", position=0))
    assert "critique" not in obs.metadata


def test_critic_enabled_populates_observation() -> None:
    """When critic_enabled=True, observation.metadata.critique is populated."""
    config = CurriculumConfig(critic_enabled=True)
    env = DrugDiscoveryEnvironment(seed=42, config=config)
    env.reset(difficulty="trivial")
    obs = env.step(MoleculeAction(action_type="ADD_FRAGMENT", fragment="C", position=0))
    assert "critique" in obs.metadata
    critique = obs.metadata["critique"]
    assert "overall" in critique
    assert critique["overall"] in ("approve", "revise", "reject")
    assert "issues" in critique
    assert "summary" in critique
    assert critique["summary"].startswith("Critic says:")
