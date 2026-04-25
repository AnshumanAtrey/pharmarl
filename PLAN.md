# PharmaRL — Build Plan (Stage 1: DRD2 — canonical molRL benchmark)

Team **AI Mafias** — Round 2 Grand Finale, Apr 25-26, 2026 (~24-27hr to deadline 5 PM Apr 26).

---

## What this env is

A multi-step OpenEnv environment where an LLM agent designs drug-like molecules by iteratively editing SELFIES strings. State changes with every edit; reward is computed by Therapeutics Data Commons (TDC) oracles plus RDKit validity checks.

**Stage 1 (default)**: binding component = **DRD2** (canonical MOSES/GuacaMol molecular RL benchmark — every molecular RL paper from MolDQN through REINVENT, GraphAF, GFlowNets benchmarks on this). Plus QED + SA + CYP3A4 toxicity. No native chemistry deps required.

The novelty in PharmaRL is the **policy class**: most molecular RL uses GNN/RNN policies trained from scratch. PharmaRL is the first OpenEnv-native env where an LLM acts as the policy — chat-agent SELFIES editing — against the canonical benchmark.

**Stage 2 (opt-in extension)**: enable real clinical docking targets via `PHARMARL_ENABLE_DOCKING=1`:
- `7l11_docking_normalize` — SARS-CoV-2 NSP15 (real COVID antiviral target)
- `2rgp_docking_normalize` — EGFR T790M (drug-resistant lung cancer)
- `1iep_docking_normalize` — ABL kinase (CML, imatinib's target)
- `4rlu_docking_normalize` — β-secretase 1 (Alzheimer's)

Stage 2 requires pyscreener + OpenBabel + AutoDock Vina installed on the deploy host. The env code probes for these; if absent, falls back to Stage 1 cleanly. The observation's `target` field always reflects the *actual* active oracle — never a marketing name.

## Episode mechanics

```
reset(episode_id) → starting molecule (depends on difficulty)
  step 0: state = {molecule: "C1CCNCC1", target: "DRD2_dopamine_D2_receptor", step: 0, reward: 0}
  step 1..N: agent issues ADD_FRAGMENT | REMOVE_FRAGMENT | SUBSTITUTE_ATOM | TERMINATE
            env validates → mutates SELFIES → returns new state + reward
  episode ends: TERMINATE action OR max_steps reached
final reward: composite (binding + QED + SA - toxicity), scaled 0-10
```

State is keyed by `episode_id` so 8 parallel GRPO rollouts (group size G=8) each get a persistent env without colliding.

## Reward structure

| Component | Source | Weight | Range |
|-----------|--------|--------|-------|
| Binding affinity | TDC DRD2 classifier (Stage 1) or pyscreener docking (Stage 2) | 0.40 | 0..1 (normalized from -12..0 kcal/mol when docking) |
| Drug-likeness | RDKit QED | 0.25 | 0..1 |
| Synthesizability | RDKit SA score | 0.15 | inverted from 1..10 |
| Toxicity penalty | TDC CYP3A4 | 0.20 | 0..1 (inverted) |
| Lipinski shaping | step-wise +0.05 if passes | dense | per-step bonus |
| Parse penalty | -0.5 on malformed action JSON | dense | discourages format errors |
| Lipinski gate | composite × 0.5 if final molecule fails Rule of 5 | terminal | anti-reward-hacking |

## Curriculum (3 tiers)

| Tier | Start molecule | Reward components | Action vocab | Max steps |
|------|----------------|---------------------|--------------|-----------|
| Trivial (steps 0-100) | DRD2-friendly scaffold (piperidine etc.) | QED only | 5 fragments | 10 |
| Easy (steps 100-300) | Smaller scaffold | QED + binding | 15 fragments | 15 |
| Hard (steps 300+) | Single carbon | All 4 components | 50 fragments | 20 |

Trivial tier = insurance for the 20% Reward Improvement criterion (a moving curve in <100 GRPO steps). Hard tier = the real proof of medicinal-chemistry intuition; this is the curve we need on overnight training.

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
├── inference.py                  ← OpenAI-client agent loop (uses API → not in default flow)
│
├── server/
│   ├── app.py                    ← FastAPI wiring with session-keyed /reset and /step
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
│       ├── docking_mpro.py       ← TDC binding oracle (DRD2 default, docking on Stage 2)
│       ├── qed.py
│       ├── sa.py
│       └── toxicity.py
│
├── scripts/
│   ├── validate_stack.py         ← Gate 1: TDC + RDKit + SELFIES sanity
│   └── smoke_notebook_locally.py ← Replays the notebook's HTTP loop without an LLM (free)
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
│   ├── test_oracles.py
│   └── test_validation.py
│
└── docs/
    ├── chemistry-primer.md       ← Q&A defense (Lipinski, kcal/mol, SELFIES)
    ├── env-spec.md               ← formal state/action/reward spec
    ├── qa-defense.md             ← 30-sec answers for judge questions
    └── pitch-script.md           ← 90-sec pitch outline
```

## Validation gates

| Gate | When | Pass criteria | Failure response |
|------|------|---------------|------------------|
| 1. Stack | Hour 0 | TDC + RDKit + SELFIES installed; ≥4 oracles return numbers | Pivot |
| 2. Local server | Hour 5-6 | `uvicorn server.app:app` boots; manual episode completes | Fix env logic |
| 3. HF deploy | Hour 18-20 | `/health` returns OK on Space URL | Fix Dockerfile / port |
| 4. Training | Hour 20-22 | W&B reward curve moves on Trivial tier in 50 steps | Lower difficulty further |
| 5. Hard-tier curve | Overnight | DRD2 score moves on hard tier across 200+ steps | Honest about limitation |

## Submission checklist (4 URLs MUST be in README)

1. HF Space URL — `https://huggingface.co/spaces/anshumanatrey/pharmarl`
2. Colab notebook URL — public Colab share
3. Code repo URL — `https://github.com/AnshumanAtrey/pharmarl` ✅
4. YouTube/HF blog video URL — 90-sec pitch

## Team split (parallel tracks)

- **Anshuman**: env code (server/, models.py, oracles); pitch script; Q&A defense
- **Sahil**: training notebook (Colab + Unsloth + TRL GRPO + W&B)
- **Vijay**: deployment (Dockerfile + HF Space + README + 90s video)

## Q&A defense (be ready to answer in 30 seconds)

See `docs/qa-defense.md` for the canonical answers. Headline ones:

- **What's novel?** First OpenEnv-native env where an LLM is the policy. Most prior molRL uses GNN/RNN; we use chat-agent SELFIES editing.
- **What target are you optimizing?** DRD2 dopamine D2 receptor (CNS therapeutics). Stage 2 extends to 4 disease-relevant docking targets.
- **Why DRD2?** Canonical MOSES/GuacaMol benchmark. Recognized by every molRL paper since MolDQN. No native chemistry deps. Aspirin 0.0003, haloperidol 0.99 — clean validated signal.
- **Why SELFIES?** Always-valid. Prevents reward hacking via gibberish strings.
- **Reward hacking?** Lipinski Rule of 5 hard gate (50% reduction if final fails) + parse penalty + composite reward + invalid SELFIES auto-rejected.
- **RLVE compliance?** Procedural episode generation, 3-tier adaptive difficulty, algorithmic verification (TDC oracles, not LLM judges).
- **How does this transfer to a real antiviral?** Same env, swap the oracle. With `PHARMARL_ENABLE_DOCKING=1` on a host with Vina installed, the trained model evaluates against `7l11_docking` (NSP15) directly. The medicinal-chemistry primitives transfer.

## Anti-staleness design

The env is generative, not retrieval-based:
- 10^60 valid drug-like molecules in the search space
- GRPO samples 8 different actions per step (temperature > 0)
- Procedurally varied starting molecules per episode (random seed pool of 200)
- Curriculum cycles target/scaffold combinations
- Oracle determinism is a *feature* (verifiable reward), not staleness

The model never sees the same trajectory twice in training.
