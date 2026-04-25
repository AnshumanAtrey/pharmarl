"""Reward red-team: adversarial molecules designed to expose hacking paths.

Q57 of the hackathon guide: "Do not optimize a reward you have not tried to
break yourself first." This file hand-crafts molecules that *each* try to
exploit one axis of the composite reward, then asserts the composite still
orders them correctly.

If any of these tests fail, the reward weights or Lipinski gate need tuning
*before* GRPO runs — otherwise the agent will discover the loophole faster
than you will.
"""

from __future__ import annotations

import pytest

from server.grader import terminal_reward


# ─── Adversarial cases ────────────────────────────────────────────────────
#
# Each row: (smiles, label, expected_property)
#
# These are *deliberately* chosen to attack one dimension of the reward:
#
#   APROVED_DRUG    — should win: real drug, balanced across all 4 axes
#   QED_GAMER       — small/clean molecule that scores high on QED but
#                     contributes nothing to binding (the trivial-tier hack)
#   SA_GAMER        — methane: trivially easy to synthesize but useless
#   TOX_TRAP        — known toxic compound that might still score on QED/SA
#   COMPLEX_FAIL    — large, non-druglike, fails Lipinski (gate should bite)
#   NONSENSE        — fails RDKit parse, must score 0 across the board

CASES = [
    ("CC(=O)Oc1ccccc1C(=O)O",                     "APPROVED_DRUG (aspirin)",  "balanced"),
    ("CCO",                                        "QED_GAMER (ethanol)",     "trivial"),
    ("C",                                          "SA_GAMER (methane)",      "trivial"),
    ("Clc1ccc(C(=O)c2ccc(Cl)cc2)cc1",              "TOX_TRAP (DDT-like)",     "toxic"),
    ("CC(C)(C)CC(C)(C)CC(C)(C)CC(C)(C)CC(C)(C)C", "COMPLEX_FAIL (large alkane)", "lipinski_fail"),
    ("not_a_real_smiles_!!!",                      "NONSENSE",                "invalid"),
]


@pytest.mark.parametrize("smiles,label,_kind", CASES)
def test_terminal_reward_in_range(smiles: str, label: str, _kind: str) -> None:
    """No adversarial molecule should produce out-of-range reward."""
    tr = terminal_reward(smiles)
    assert tr.reward >= -0.1, f"{label}: negative-spike reward {tr.reward}"
    assert tr.reward <= 12.0, f"{label}: above-cap reward {tr.reward} (composite > 1?)"
    assert 0.0 <= tr.composite <= 1.05, f"{label}: composite out of [0,1.05]: {tr.composite}"


def test_approved_drug_beats_trivial_gamers() -> None:
    """A real drug must beat single-axis gamers — otherwise reward is hackable."""
    aspirin = terminal_reward("CC(=O)Oc1ccccc1C(=O)O")
    methane = terminal_reward("C")
    ethanol = terminal_reward("CCO")
    assert aspirin.reward > methane.reward, (
        f"REWARD HACK: methane (R={methane.reward:.2f}) ≥ aspirin "
        f"(R={aspirin.reward:.2f}) — composite is over-weighting QED/SA"
    )
    assert aspirin.reward > ethanol.reward, (
        f"REWARD HACK: ethanol (R={ethanol.reward:.2f}) ≥ aspirin "
        f"(R={aspirin.reward:.2f}) — small-molecule QED gaming wins"
    )


def test_lipinski_gate_punishes_failure() -> None:
    """Gate must halve composite for Lipinski-failing molecules."""
    bad = terminal_reward("CC(C)(C)CC(C)(C)CC(C)(C)CC(C)(C)CC(C)(C)C")
    # If Lipinski passes for this monster, the gate isn't checking what we think
    # If it fails, composite must be ≤ 0.5 (halved)
    if not bad.lipinski_passes:
        assert bad.composite <= 0.5 + 1e-6, (
            f"Lipinski gate didn't halve composite: {bad.composite}"
        )


def test_nonsense_smiles_safe() -> None:
    """Invalid SMILES must produce 0 reward, no crashes."""
    tr = terminal_reward("not_a_real_smiles_!!!")
    assert tr.reward == 0.0 or tr.reward == pytest.approx(0.0, abs=0.01)
    assert all(v == 0.0 or 0.0 <= v <= 1.0 for v in tr.components.values())


def test_components_visible_for_every_case() -> None:
    """Trainer relies on per-component logging — they must always be present."""
    for smiles, label, _ in CASES:
        tr = terminal_reward(smiles)
        for required in ("qed", "docking", "sa", "toxicity_raw", "toxicity_clean"):
            assert required in tr.components, (
                f"{label}: missing component {required!r} — trainer will skip W&B log"
            )


def test_curriculum_tier_isolation() -> None:
    """Trivial tier must score *only* on QED — proving the curriculum gate works."""
    aspirin = terminal_reward(
        "CC(=O)Oc1ccccc1C(=O)O", components_active=("qed",)
    )
    # Composite for trivial = QED directly
    assert aspirin.composite == pytest.approx(aspirin.components["qed"], abs=1e-6)


def test_easy_tier_is_qed_plus_docking() -> None:
    """Easy tier must combine only QED + docking, not SA/tox."""
    aspirin = terminal_reward(
        "CC(=O)Oc1ccccc1C(=O)O", components_active=("qed", "docking")
    )
    qed = aspirin.components["qed"]
    docking = aspirin.components["docking"]
    expected = (0.25 * qed + 0.40 * docking) / (0.25 + 0.40)
    # Lipinski should pass for aspirin → no halving
    if aspirin.lipinski_passes:
        assert aspirin.composite == pytest.approx(expected, abs=1e-3)
