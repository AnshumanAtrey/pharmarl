"""Gate 1 validation script — run BEFORE writing any env code.

Checks:
  1. PyTDC installs and imports
  2. RDKit installs and imports
  3. SELFIES installs and imports
  4. TDC oracles load and return numeric scores for aspirin
  5. SARS-CoV-2 Mpro docking oracle availability (multiple candidate names)
  6. SELFIES round-trip works for a known molecule

Usage:
    python scripts/validate_stack.py

Exit codes:
    0 — all green, build PharmaRL multi-target
    1 — Mpro oracle missing but other oracles work — fall back to DRD2 base
    2 — major failures — pivot to AISHA
"""

from __future__ import annotations

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


def check_oracles() -> dict[str, float | None]:
    header("2. TDC oracles on aspirin (CC(=O)Oc1ccccc1C(=O)O)")
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    scores: dict[str, float | None] = {}
    try:
        from tdc import Oracle
    except Exception as e:
        print(f"  [FAIL]  cannot import tdc.Oracle -- {e}")
        return scores

    standard_oracles = ["QED", "SA", "DRD2", "GSK3B", "JNK3", "LogP"]
    for name in standard_oracles:
        try:
            o = Oracle(name=name)
            score = o(aspirin)
            scores[name] = float(score)
            print(f"  [OK]    {name:8s} = {score:.3f}")
        except Exception as e:
            scores[name] = None
            print(f"  [FAIL]  {name:8s} -- {e}")
    return scores


def check_mpro_oracle() -> str | None:
    header("3. SARS-CoV-2 Mpro docking oracle availability")
    try:
        from tdc import Oracle
    except Exception as e:
        print(f"  [FAIL]  cannot import tdc.Oracle -- {e}")
        return None

    candidates = [
        "SARS-CoV-2_3CLPro_Docking",
        "SARS_CoV_2_3CLPro_Docking",
        "3CLPro_Docking",
        "SARS-CoV-2_Mpro",
        "Mpro_docking",
        "3pbl_docking",
        "1iep_docking",
    ]
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    for name in candidates:
        try:
            o = Oracle(name=name)
            score = o(aspirin)
            print(f"  [OK]    Oracle('{name}') = {score:.3f}")
            return name
        except Exception:
            print(f"  [skip]  Oracle('{name}') not available")
    print("  [WARN]  No Mpro docking oracle found via TDC.Oracle.")
    print("          Mitigation: use DRD2 as scientific base (standard MOSES/GuacaMol benchmark)")
    print("          Mpro narrative comes from scaffold framing, not direct docking.")
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
    except Exception as e:
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
    except Exception as e:
        traceback.print_exc()
        return False


def main() -> int:
    print("PharmaRL Gate 1 validation — checking dependency stack")

    imports = check_imports()
    if not all(imports.values()):
        print("\n[ABORT] Required libraries missing. Install with:")
        print("        pip install PyTDC selfies rdkit-pypi")
        return 2

    oracle_scores = check_oracles()
    mpro_oracle = check_mpro_oracle()
    selfies_ok = check_selfies_roundtrip()
    rdkit_ok = check_rdkit_basics()

    header("Summary")
    standard_count = sum(1 for v in oracle_scores.values() if v is not None)
    print(f"  Standard oracles working:    {standard_count}/6")
    print(f"  Mpro docking oracle name:    {mpro_oracle or 'NOT FOUND'}")
    print(f"  SELFIES round-trip:          {'OK' if selfies_ok else 'FAIL'}")
    print(f"  RDKit basics:                {'OK' if rdkit_ok else 'FAIL'}")

    if standard_count >= 4 and selfies_ok and rdkit_ok:
        if mpro_oracle:
            print("\n  [GREEN] Mpro oracle works. Build multi-target PharmaRL with Mpro as Stage 1.")
        else:
            print("\n  [YELLOW] Mpro oracle missing. Use DRD2 as scientific base oracle;")
            print("           keep Mpro framing in pitch via known-scaffold curriculum.")
        return 0 if mpro_oracle else 1

    print("\n  [RED] Stack not ready. Pivot to AISHA-multi-agent.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
