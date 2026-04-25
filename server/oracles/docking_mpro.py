"""Binding-activity oracle for PharmaRL — supports per-target scoring.

Stage 1 default: **DRD2** — the canonical MOSES/GuacaMol molecular RL
benchmark. Every paper from MolDQN (2019) through REINVENT, GraphAF, and
the 2024 GFlowNet variants reports on DRD2.

Multi-target support (Stage 1 classifiers, no native deps):
    DRD2   — dopamine D2 receptor (GPCR; CNS therapeutics)
    GSK3B  — glycogen synthase kinase 3-β (Ser/Thr kinase; Alzheimer's-relevant)
    JNK3   — c-Jun N-terminal kinase 3 (MAP kinase; neurodegeneration)

The held-out generalization test trains on {DRD2, GSK3B} and evaluates on JNK3.

Stage 2 extensions (requires pyscreener + AutoDock Vina + OpenBabel
installed on the deploy host):
    7l11_docking_normalize  → SARS-CoV-2 NSP15
    2rgp_docking_normalize  → EGFR T790M (drug-resistant lung cancer)
    1iep_docking_normalize  → ABL kinase (CML — imatinib's target)
    4rlu_docking_normalize  → β-secretase 1 (Alzheimer's)

When pyscreener is available the docking oracles take precedence; otherwise
classifier oracles are loaded directly and contribute the binding component.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from rdkit import Chem

logger = logging.getLogger(__name__)

# Set PHARMARL_ENABLE_DOCKING=1 to probe pyscreener-backed docking oracles
# (requires OpenBabel + AutoDock Vina installed on the host). Off by default
# because the probe adds ~20s of Ray-instance startup before falling back.
_DOCKING_ENABLED = os.environ.get("PHARMARL_ENABLE_DOCKING", "").lower() in ("1", "true", "yes")

# Single-oracle cache (the legacy default path — used when /reset doesn't pass target)
_ORACLE = None
_ORACLE_NAME: Optional[str] = None
_TRIED_INIT = False

# Per-target classifier cache (the multi-target path)
_TARGET_CACHE: Dict[str, Any] = {}
_TARGET_TRIED: Dict[str, bool] = {}

# Probed in order — first one that loads wins.
_DOCKING_CANDIDATES_PYSCREENER = [
    "7l11_docking_normalize",   # SARS-CoV-2 NSP15
    "2rgp_docking_normalize",   # EGFR T790M
    "1iep_docking_normalize",   # ABL kinase
    "4rlu_docking_normalize",   # β-secretase 1
]
_CLASSIFIER_FALLBACKS = [
    "DRD2",                     # canonical MOSES/GuacaMol benchmark
    "GSK3B",                    # Alzheimer's-related kinase
    "JNK3",                     # neurodegeneration kinase
]

# Targets the env knows how to route to — short names accepted by /reset.
KNOWN_TARGETS = ("DRD2", "GSK3B", "JNK3", "AMLODIPINE_MPO")
DEFAULT_TARGET = "DRD2"

# Maps short target names to TDC Oracle registry names. Most targets share
# the short form (DRD2 → "drd2"); the cross-family secondary held-out maps
# to a TDC MPO oracle that scores similarity-to-amlodipine + drug-likeness.
# Amlodipine is an L-type calcium channel blocker — orthogonal pharmacology
# to the Ser/Thr + MAP kinases used for primary training and held-out.
_TDC_ORACLE_NAME = {
    "DRD2": "DRD2",
    "GSK3B": "GSK3B",
    "JNK3": "JNK3",
    "AMLODIPINE_MPO": "amlodipine_mpo",
}


def _lazy_init() -> None:
    """Loads the default single-oracle path (Stage 1 = DRD2 unless docking enabled)."""
    global _ORACLE, _ORACLE_NAME, _TRIED_INIT
    if _TRIED_INIT:
        return
    _TRIED_INIT = True

    try:
        from tdc import Oracle  # type: ignore
    except ImportError:
        logger.warning("PyTDC not installed — binding oracle disabled.")
        return

    if _DOCKING_ENABLED:
        for name in _DOCKING_CANDIDATES_PYSCREENER:
            try:
                o = Oracle(name=name)
                o("CCO")
                _ORACLE = o
                _ORACLE_NAME = name
                logger.info("Loaded docking oracle: %s", name)
                return
            except Exception:
                continue
        logger.info(
            "PHARMARL_ENABLE_DOCKING=1 set but no docking oracle loaded "
            "(pyscreener/Vina/OpenBabel missing). Falling back to classifier."
        )

    for name in _CLASSIFIER_FALLBACKS:
        try:
            _ORACLE = Oracle(name=name)
            _ORACLE_NAME = name
            logger.info(
                "Using classifier oracle %s for binding component "
                "(docking unavailable — install pyscreener+Vina to enable docking).",
                name,
            )
            return
        except Exception:
            continue

    logger.error("No binding oracle available — scores will be 0.0")


def _load_target_oracle(target: str) -> Any:
    """Lazy-load and cache a TDC classifier oracle for a specific short target name."""
    if target in _TARGET_CACHE:
        return _TARGET_CACHE[target]
    if _TARGET_TRIED.get(target):
        return None
    _TARGET_TRIED[target] = True

    if target not in KNOWN_TARGETS:
        logger.warning("Unknown target %r — must be one of %s", target, KNOWN_TARGETS)
        return None

    try:
        from tdc import Oracle  # type: ignore
    except ImportError:
        logger.warning("PyTDC not installed — target %s disabled.", target)
        return None

    tdc_name = _TDC_ORACLE_NAME.get(target, target)
    try:
        oracle = Oracle(name=tdc_name)
        _TARGET_CACHE[target] = oracle
        logger.info("Loaded per-target oracle %s (TDC=%s).", target, tdc_name)
        return oracle
    except Exception as e:
        logger.warning("Failed to load oracle for target %s: %s", target, e)
        return None


def _normalize_score(raw: float) -> float:
    """Normalize raw oracle output to [0, 1] (1 = best binder)."""
    if raw < 0:
        # Docking score in kcal/mol — clamp to [-12, 0], invert
        return float(max(0.0, min(1.0, -raw / 12.0)))
    # Classifier — already in [0, 1]
    return float(max(0.0, min(1.0, raw)))


def score_mpro_docking(smiles: str, target: Optional[str] = None) -> float:
    """Returns binding score in [0, 1] (higher = stronger binder).

    If `target` is None, uses the legacy single-oracle path (Stage 1 default = DRD2).
    If `target` is one of KNOWN_TARGETS, routes to that classifier specifically —
    enables multi-target training and held-out evaluation.

    Raw oracle output convention varies:
      - Docking score: kcal/mol, e.g. -8.0 (strong) to -2.0 (weak) → normalize
      - DRD2/kinase classifier: probability 0..1 → use directly
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return 0.0

    if target is not None:
        oracle = _load_target_oracle(target)
        if oracle is None:
            return 0.0
        try:
            raw = float(oracle(smiles))
        except Exception as e:
            logger.debug("Oracle %s eval failed for %s: %s", target, smiles, e)
            return 0.0
        return _normalize_score(raw)

    # Legacy single-oracle path
    _lazy_init()
    if _ORACLE is None:
        return 0.0
    try:
        raw = float(_ORACLE(smiles))
    except Exception as e:
        logger.debug("Oracle eval failed for %s: %s", smiles, e)
        return 0.0
    return _normalize_score(raw)


def get_active_oracle_name() -> Optional[str]:
    """Name of the default single-target oracle (used when /reset omits target)."""
    _lazy_init()
    return _ORACLE_NAME


_ORACLE_TO_TARGET_NAME = {
    "DRD2": "DRD2_dopamine_D2_receptor",
    "GSK3B": "GSK3B_glycogen_synthase_kinase_3_beta",
    "JNK3": "JNK3_c-Jun_N-terminal_kinase_3",
    "AMLODIPINE_MPO": "amlodipine_MPO_L-type_calcium_channel_proxy",
    "7l11_docking_normalize": "SARS-CoV-2_NSP15_endoribonuclease",
    "2rgp_docking_normalize": "EGFR_T790M_kinase",
    "1iep_docking_normalize": "ABL_kinase",
    "4rlu_docking_normalize": "BACE1_beta_secretase",
}


def get_target_full_name(target_short: Optional[str] = None) -> str:
    """Map a short target name (e.g. 'DRD2') to the human-readable full name.

    With no argument, returns the active default-oracle's full name (legacy path).
    """
    if target_short is not None:
        return _ORACLE_TO_TARGET_NAME.get(target_short, target_short)
    name = get_active_oracle_name()
    if name is None:
        return "no_oracle_active"
    return _ORACLE_TO_TARGET_NAME.get(name, name)


# Backwards-compatible alias for callers expecting the old name
get_active_target_name = get_target_full_name
