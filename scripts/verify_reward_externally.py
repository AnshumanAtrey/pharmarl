"""Reproduce PharmaRL's reward signal using only TDC + RDKit — no PharmaRL imports.

The point of this script is reviewer trust. PharmaRL does not implement its own
oracle scoring. We wrap Therapeutics Data Commons (TDC) — Huang et al.,
*Nature Chemical Biology* 2022 — which is the same scoring infrastructure used
in REINVENT (2017), MolDQN (2019), GraphAF (2020), GFlowNets (2021), MOSES,
and GuacaMol.

If our reward function were a self-graded heuristic, this script would not
exist. Because the reward is third-party, anyone can run TDC directly and
get the same numbers. That is what this script demonstrates.

Run:
    pip install pytdc rdkit
    python scripts/verify_reward_externally.py

Expected (within classifier float tolerance):
    aspirin     DRD2 ~ 0.0003   QED ~ 0.55   SA ~ 1.58
    haloperidol DRD2 ~ 1.00     QED ~ 0.76   SA ~ 2.12
    ibuprofen   DRD2 ~ 0.002    QED ~ 0.82   SA ~ 2.19

Note: SA from TDC is the raw Ertl/Schuffenhauer score in [1, 10] (lower =
easier to synthesize). PharmaRL's server normalizes this to [0, 1] before
composing the terminal reward (see server/oracles/sa.py). Compare these
raw TDC values against the classifier outputs in PharmaRL's observation
metadata; the composite reward in our server applies the documented
weights + Lipinski gate.

If the DRD2 numbers above diverge from what PharmaRL's /step returns for
the same SMILES on TERMINATE, the bug is on our side, not TDC's.
"""

from __future__ import annotations

import sys


# Reference molecules — well-known structures from the medicinal-chemistry
# literature. Their TDC scores are stable across pytdc versions; we use them
# as ground truth in tests/test_reward_redteam.py too (haloperidol > 0.9 on
# DRD2 is a regression guard against silently corrupted classifier weights).
REFERENCE_SMILES = [
    ("aspirin",     "CC(=O)Oc1ccccc1C(=O)O"),
    ("haloperidol", "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1"),
    ("ibuprofen",   "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
    ("caffeine",    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
]


def _try_import():
    try:
        from tdc import Oracle
    except ImportError:
        print(
            "ERROR: pytdc is not installed. Run `pip install pytdc` and try again.",
            file=sys.stderr,
        )
        sys.exit(2)
    return Oracle


def main() -> int:
    Oracle = _try_import()

    # All four oracles below come from TDC's published Oracle registry.
    # No PharmaRL code is involved in producing these numbers.
    print("Loading TDC oracles (DRD2, QED, SA, GSK3B, JNK3, CYP3A4_Veith)...")
    oracles = {
        "DRD2":   Oracle(name="DRD2"),
        "GSK3B":  Oracle(name="GSK3B"),
        "JNK3":   Oracle(name="JNK3"),
        "QED":    Oracle(name="QED"),
        "SA":     Oracle(name="SA"),
        "CYP3A4": Oracle(name="CYP3A4_Veith"),
    }

    print()
    print(f"{'molecule':<12} {'DRD2':>8} {'GSK3B':>8} {'JNK3':>8} {'QED':>8} {'SA':>8} {'CYP3A4':>8}")
    print("-" * 70)

    for name, smi in REFERENCE_SMILES:
        scores = {}
        for k, o in oracles.items():
            try:
                scores[k] = float(o(smi))
            except Exception as e:  # noqa: BLE001
                print(f"  {name}: oracle {k} failed → {e}", file=sys.stderr)
                scores[k] = float("nan")
        print(
            f"{name:<12} "
            f"{scores['DRD2']:>8.4f} "
            f"{scores['GSK3B']:>8.4f} "
            f"{scores['JNK3']:>8.4f} "
            f"{scores['QED']:>8.4f} "
            f"{scores['SA']:>8.4f} "
            f"{scores['CYP3A4']:>8.4f}"
        )

    print()
    print("These numbers are TDC's, not ours. PharmaRL composes them with the")
    print("weights documented in server/grader.py and applies the Lipinski")
    print("terminal gate, but the underlying signal is third-party.")
    print()
    print("Sanity check: haloperidol (a known D2 antagonist) should score >0.9")
    print("on DRD2; aspirin should score near 0. If either of those is wrong,")
    print("your local TDC weights are stale — re-fetch them.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
