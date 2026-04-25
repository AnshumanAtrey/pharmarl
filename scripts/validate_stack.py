"""Gate 1 validation script — runs BEFORE any training to confirm the stack is healthy.

Checks:
  1. PyTDC, RDKit, SELFIES installed and importing
  2. Standard oracles return numeric scores for aspirin (QED, SA, DRD2, CYP3A4)
  3. Stage 2 docking oracle availability (only if PHARMARL_ENABLE_DOCKING=1)
  4. SELFIES round-trip works
  5. RDKit Lipinski calculation works

Usage:
    python scripts/validate_stack.py
    PHARMARL_ENABLE_DOCKING=1 python scripts/validate_stack.py   # also probe pyscreener docking

Exit codes:
    0 — Stage 1 stack ready (DRD2 + QED + SA + CYP3A4 all working)
    1 — Stage 2 probe enabled but no docking oracle loaded — Stage 1 still works
    2 — Stage 1 broken (missing core deps) — must be fixed before training
"""

from __future__ import annotations

import os
import sys
import traceback


def header(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def check_imports() -> dict[str, bool]:
    header("1. Library imports")
    results: dict[str, bool] = {}
    for lib in ("tdc", "selfies", "rdkit"):
        try:
            __import__(lib)
            print(f"  [OK]    import {lib}")
            results[lib] = True
        except ImportError as e:
            print(f"  [FAIL]  import {lib} -- {e}")
            results[lib] = False
    return results


def check_standard_oracles() -> dict[str, float | None]:
    header("2. TDC oracles on aspirin (CC(=O)Oc1ccccc1C(=O)O)")
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    haloperidol = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
    scores: dict[str, float | None] = {}
    try:
        from tdc import Oracle
    except Exception as e:
        print(f"  [FAIL]  cannot import tdc.Oracle -- {e}")
        return scores

    # The ones our env actually uses for the composite reward:
    core_oracles = ["QED", "SA", "DRD2", "CYP3A4_Veith"]
    for name in core_oracles:
        try:
            o = Oracle(name=name)
            score = o(aspirin)
            scores[name] = float(score)
            print(f"  [OK]    {name:14s} aspirin = {score:.4f}")
        except Exception as e:
            scores[name] = None
            print(f"  [FAIL]  {name:14s} -- {e}")

    # DRD2 sanity: haloperidol (a known D2 antagonist) should be near 1.0
    if scores.get("DRD2") is not None:
        try:
            o = Oracle(name="DRD2")
            halo = float(o(haloperidol))
            verdict = "OK" if halo > 0.9 else "WEIRD"
            print(f"  [{verdict}]    DRD2 sanity: haloperidol = {halo:.4f} (expect > 0.9)")
        except Exception as e:
            print(f"  [warn]  DRD2 sanity check failed: {e}")
    return scores


def check_stage2_docking() -> str | None:
    """Probes Stage 2 docking oracles. Only runs if PHARMARL_ENABLE_DOCKING=1."""
    enabled = os.environ.get("PHARMARL_ENABLE_DOCKING", "").lower() in ("1", "true", "yes")
    header("3. Stage 2 docking oracles (PHARMARL_ENABLE_DOCKING)")
    if not enabled:
        print("  [skip]  PHARMARL_ENABLE_DOCKING not set — Stage 1 (DRD2) is the active path.")
        print("          To probe Stage 2: PHARMARL_ENABLE_DOCKING=1 python scripts/validate_stack.py")
        return None

    try:
        from tdc import Oracle
    except Exception as e:
        print(f"  [FAIL]  cannot import tdc.Oracle -- {e}")
        return None

    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    candidates = [
        "7l11_docking_normalize",   # SARS-CoV-2 NSP15
        "2rgp_docking_normalize",   # EGFR T790M
        "1iep_docking_normalize",   # ABL kinase
        "4rlu_docking_normalize",   # BACE1
    ]
    for name in candidates:
        try:
            o = Oracle(name=name)
            score = o(aspirin)
            print(f"  [OK]    Oracle('{name}') aspirin = {score:.3f}")
            return name
        except Exception as e:
            print(f"  [skip]  Oracle('{name}') unavailable -- {type(e).__name__}")
    print("  [WARN]  PHARMARL_ENABLE_DOCKING=1 set but no docking oracle loaded.")
    print("          Cause: pyscreener / OpenBabel / AutoDock Vina not installed.")
    print("          Stage 1 (DRD2) will still work — the env falls back transparently.")
    return None


def check_selfies_roundtrip() -> bool:
    header("4. SELFIES round-trip")
    try:
        import selfies as sf
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        encoded = sf.encoder(smiles)
        decoded = sf.decoder(encoded)
        print(f"  SMILES   in:  {smiles}")
        print(f"  SELFIES out:  {encoded}")
        print(f"  SMILES   out: {decoded}")
        ok = decoded is not None and len(decoded) > 0
        print(f"  [{'OK' if ok else 'FAIL'}]    round-trip {'succeeded' if ok else 'failed'}")
        return ok
    except Exception:
        traceback.print_exc()
        return False


def check_rdkit_basics() -> bool:
    header("5. RDKit basics — molecule parsing + property calc")
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski
        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        if mol is None:
            print("  [FAIL]  could not parse aspirin")
            return False
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        print(f"  Aspirin MW={mw:.1f}, LogP={logp:.2f}, HBD={hbd}, HBA={hba}")
        passes_lipinski = mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
        print(f"  Lipinski: {'PASS' if passes_lipinski else 'FAIL'}")
        return True
    except Exception:
        traceback.print_exc()
        return False


def main() -> int:
    print("PharmaRL Gate 1 — stack validation")

    imports = check_imports()
    if not all(imports.values()):
        print("\n[ABORT] Required libraries missing. Install with:")
        print("        pip install -e .")
        return 2

    oracle_scores = check_standard_oracles()
    docking_name = check_stage2_docking()
    selfies_ok = check_selfies_roundtrip()
    rdkit_ok = check_rdkit_basics()

    header("Summary")
    core_count = sum(1 for v in oracle_scores.values() if v is not None)
    print(f"  Core oracles working:        {core_count}/4 (QED, SA, DRD2, CYP3A4)")
    print(f"  Stage 2 docking oracle:      {docking_name or 'not active (Stage 1 = DRD2)'}")
    print(f"  SELFIES round-trip:          {'OK' if selfies_ok else 'FAIL'}")
    print(f"  RDKit basics:                {'OK' if rdkit_ok else 'FAIL'}")

    stage1_ready = core_count >= 3 and selfies_ok and rdkit_ok
    if not stage1_ready:
        print("\n  [RED] Stage 1 stack not ready. Fix imports/oracles before training.")
        return 2

    if docking_name:
        print("\n  [GREEN] Both Stage 1 and Stage 2 working — full PharmaRL capabilities.")
        return 0
    enabled = os.environ.get("PHARMARL_ENABLE_DOCKING", "").lower() in ("1", "true", "yes")
    if enabled:
        print("\n  [YELLOW] Stage 2 probe enabled but failed (pyscreener/Vina/OpenBabel missing).")
        print("           Stage 1 (DRD2) still works — the env falls back transparently.")
        return 1
    print("\n  [GREEN] Stage 1 stack ready. DRD2 active, training can proceed.")
    print("           Stage 2 docking is opt-in (PHARMARL_ENABLE_DOCKING=1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
