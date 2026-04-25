# PharmaRL — Build Plan (Stage 1: SARS-CoV-2 Mpro)

Team **AI Mafias** — Round 2 Grand Finale, Apr 25-26, 2026 (~27hr to deadline 5 PM Apr 26).

---

## What this env is

A multi-step OpenEnv environment where an LLM agent designs drug-like molecules by iteratively editing SELFIES strings. State changes with every edit; reward is computed by Therapeutics Data Commons (TDC) oracles plus RDKit validity checks.

**Stage 1 (default)**: binding component = **DRD2** (canonical MOSES/GuacaMol molecular RL benchmark — every molecular RL paper from MolDQN to today benchmarks on this). Plus QED + SA + CYP3A4 toxicity. No native chemistry deps required.

**Stage 2 (opt-in extension)**: enable real clinical docking targets via `PHARMARL_ENABLE_DOCKING=1`:
- `7l11_docking_normalize` — SARS-CoV-2 NSP15 (real COVID antiviral target, replaces Mpro narrative)
- `2rgp_docking_normalize` — EGFR T790M (drug-resistant lung cancer)
- `1iep_docking_normalize` — ABL kinase (CML, imatinib's target)
- `4rlu_docking_normalize` — β-secretase 1 (Alzheimer's)

Stage 2 requires pyscreener + OpenBabel + AutoDock Vina installed on the deploy host. The env code probes for these; if absent, falls back to Stage 1 cleanly.

## Episode mechanics

```
reset() → starting molecule (depends on difficulty)
  step 0: state = {molecule: "C", target: "Mpro", step: 0, reward: 0}
  step 1..N: agent issues ADD_FRAGMENT | REMOVE_FRAGMENT | SUBSTITUTE_ATOM | TERMINATE
            env validates → mutates SELFIES → returns new state + reward
  episode ends: TERMINATE action OR max_steps reached
final reward: composite (binding + QED + SA - toxicity), scaled 0-10
```

## Reward structure

| Component | Source | Weight | Range |
|-----------|--------|--------|-------|
| Binding affinity | TDC Mpro docking surrogate (or DRD2 fallback) | 0.40 | -12..0 kcal/mol → normalized 0..1 |
| Drug-likeness | RDKit QED | 0.25 | 0..1 |
| Synthesizability | RDKit SA score | 0.15 | inverted from 1..10 |
| Toxicity penalty | TDC CYP3A4 | 0.20 | 0..1 (inverted) |
| Lipinski shaping | step-wise +0.05 if passes | dense | per-step bonus |
| Parse penalty | -0.5 on malformed action JSON | dense | discourages format errors |

## Curriculum (3 tiers)

| Tier | Start molecule | Reward components | Action vocab | Max steps |
|------|----------------|---------------------|--------------|-----------|
| Trivial (steps 0-100) | Known nirmatrelvir-like scaffold | QED only | 5 fragments | 10 |
| Easy (steps 100-300) | Smaller scaffold | QED + binding | 15 fragments | 15 |
| Hard (steps 300+) | Single carbon | All 4 components | 50 fragments | 20 |

Trivial tier guarantees a moving reward curve (insurance for the 20% Reward Improvement criterion).

## File map

```
round2/                           ← will be renamed to pharmarl/ at end
├── .gitignore                    ← excludes research/ resources/ wandb/ etc
├── README.md                     ← submission URLs go here
├── PLAN.md                       ← this file
├── pyproject.toml
├── openenv.yaml                  ← OpenEnv manifest
├── __init__.py
├── client.py                     ← OpenEnv client wrapper
├── models.py                     ← MoleculeAction/Observation/State
├── inference.py                  ← baseline OpenAI-client agent loop
│
├── server/
│   ├── app.py                    ← FastAPI wiring
│   ├── drug_discovery_environment.py   ← reset/step (THE CORE)
│   ├── grader.py                 ← composite reward
│   ├── curriculum.py             ← 3-tier difficulty
│   ├── scenarios.py              ← starting scaffolds
│   ├── requirements.txt          ← server-only deps
│   ├── Dockerfile                ← HF Space container
│   ├── molecule_engine/
│   │   ├── fragments.py          ← 50-fragment vocabulary
│   │   ├── mutations.py          ← ADD/REMOVE/SUBSTITUTE
│   │   └── validation.py         ← Lipinski check
│   └── oracles/
│       ├── docking_mpro.py       ← TDC Mpro (DRD2 fallback)
│       ├── qed.py
│       ├── sa.py
│       └── toxicity.py
│
├── scripts/
│   └── validate_stack.py         ← Gate 1: TDC + RDKit + SELFIES sanity
│
├── colab/
│   └── train_pharmarl.ipynb      ← Unsloth + TRL GRPO notebook
│
├── examples/
│   ├── manual_episode.py         ← run one episode by hand
│   └── before_after_demo.py      ← base vs trained for video
│
├── tests/
│   ├── test_env.py
│   └── test_oracles.py
│
└── docs/
    ├── chemistry-primer.md       ← Q&A defense (Lipinski, kcal/mol, SELFIES, Mpro)
    └── env-spec.md               ← formal state/action/reward spec
```

## Validation gates

| Gate | When | Pass criteria | Failure response |
|------|------|---------------|------------------|
| 1. Stack | Hour 0 | TDC + RDKit + SELFIES installed; ≥4 oracles return numbers | Pivot to AISHA |
| 2. Local server | Hour 5-6 | `uvicorn server.app:app` boots; manual episode completes | Fix env logic |
| 3. HF deploy | Hour 7 | `/health` returns OK on Space URL | Fix Dockerfile / port |
| 4. Training | Hour 10 | W&B reward curve moves on Trivial tier in 50 steps | Lower difficulty further |

## Submission checklist (4 URLs MUST be in README)

1. HF Space URL — `https://huggingface.co/spaces/anshumanatrey/pharmarl`
2. Colab notebook URL — public Colab share
3. Code repo URL — `https://github.com/anshumanatrey/pharmarl`
4. YouTube/HF blog video URL — 90-sec pitch

## Team split (parallel tracks)

- **Anshuman**: env code (server/, models.py, oracles); pitch story
- **Sahil**: training notebook (Colab + Unsloth + TRL GRPO + W&B)
- **Vijay**: deployment (Dockerfile + HF Space + README + 90s video)

## Hour-by-hour budget

| Hours | Phase | Owner |
|-------|-------|-------|
| 0-1 | Gate 1 + scaffold | A |
| 1-3 | Core env (models + step logic + molecule_engine) | A |
| 3-4 | Oracles + grader | A |
| 4-5 | Curriculum + scenarios | A |
| 5-6 | FastAPI wiring + Gate 2 | A |
| 6-7 | Dockerfile + HF Space deploy + Gate 3 | V |
| 7-10 | Colab training notebook + Gate 4 (50-step smoke) | S |
| 10-18 | Full training run (overnight) | (auto) |
| 18-22 | Demo assets (before/after, video, README) | V |
| 22-24 | Stage 2 multi-target IF MVP shipped | A+S |
| 24-27 | Polish + Mentor Round 3 + submission | all |

## Q&A defense (be ready to answer)

- **Why SELFIES?** Always chemically valid → no reward hacking via gibberish strings.
- **Why these 4 oracles?** Independent, non-gameable, validated; TDC = Harvard/MIT.
- **Why this curriculum?** Stage 0 trivial guarantees moving curve; Stage 2 hard demonstrates real capability gain.
- **Why Mpro?** SARS-CoV-2 main protease, druggable target of nirmatrelvir (Paxlovid). India/COVID relevance for Sarvam judge.
- **Reward hacking?** Lipinski Rule of 5 hard penalty + composite reward + invalid SELFIES auto-rejected.
- **RLVE compliance?** Procedural episode generation (random scaffold seeds), adjustable difficulty (3 tiers), algorithmic verification (oracle scores).

## Anti-staleness design (the question that got us blocked)

The env is generative, not retrieval-based:
- 10^60 valid drug-like molecules in the search space
- GRPO samples 8 different actions per step (temperature > 0)
- Procedurally varied starting molecules per episode (random seed pool of 200)
- Curriculum cycles target/scaffold combinations
- Oracle determinism is a *feature* (verifiable reward), not staleness

The model never sees the same trajectory twice in training.
