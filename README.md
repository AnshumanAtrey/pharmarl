---
title: PharmaRL
emoji: 💊
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
short_description: OpenEnv-native LLM-as-policy molecular optimization.
---

# PharmaRL

**The first OpenEnv-native, LLM-as-policy environment for iterative molecular optimization.**

Most molecular RL systems (MolDQN, REINVENT, GraphAF, GFlowNets) use GNN/RNN policies trained from scratch. PharmaRL flips that: a chat agent edits SELFIES strings step-by-step, scored by Therapeutics Data Commons (TDC) oracles plus RDKit validity checks. Same canonical benchmark, new policy class.

**Stage 1 (default, what trains in minutes on Colab T4)** — binding component is the **DRD2** classifier (dopamine D2 receptor), the canonical MOSES/GuacaMol molecular RL benchmark used in nearly every published molRL paper. Plus QED + SA + CYP3A4 toxicity. No native chemistry deps required. Aspirin scores 0.0003, haloperidol 0.99 — clean signal.

**Stage 2 (opt-in, deploy-time extension)** — set `PHARMARL_ENABLE_DOCKING=1` with pyscreener + AutoDock Vina + OpenBabel installed on the host, and the binding component falls forward to one of:
- `7l11_docking_normalize` — SARS-CoV-2 NSP15 (real COVID antiviral target)
- `2rgp_docking_normalize` — EGFR T790M (drug-resistant lung cancer)
- `1iep_docking_normalize` — ABL kinase (chronic myeloid leukemia)
- `4rlu_docking_normalize` — β-secretase 1 (Alzheimer's)

Stage 2 requires a chemistry stack that does not pip-install cleanly in stock HF Space containers. The env probes for these and falls back to Stage 1 transparently — every observation reports the *actual* target being scored, never a marketing label.

> Built for Meta PyTorch OpenEnv Hackathon Apr '26 by **AI Mafias** (Anshuman, Sahil, Vijay).

---

## Submission URLs

| Resource | URL |
|----------|-----|
| Code repository | `https://github.com/AnshumanAtrey/pharmarl` |
| HF Space (deployed env) | `<TODO>` |
| Colab notebook (training) | `<TODO>` |
| Pitch video (90s) | `<TODO>` |
| W&B training run | `<TODO>` |
| Trained model on HF Hub | `<TODO>` |

---

## What this is

A **Type B** OpenEnv environment (real RL with state dynamics, not text grader). Each episode is a 10-20 step trajectory of molecular edits; each edit changes what's possible in the next step. Reward is sparse (terminal composite oracle score) plus dense shaping (Lipinski compliance per step). The agent learns medicinal-chemistry primitives — fragment selection, scaffold construction, Lipinski compliance — that transfer between targets.

```
Episode trajectory (15 steps, trivial scaffold start):
  step 0:  C1CCNCC1                  → reward 0.0   (piperidine seed)
  step 1:  CC1CCNCC1                 → reward 0.05  (Lipinski OK, +shaping)
  step 5:  c1ccc(N2CCNCC2)cc1        → reward 0.05
  step 14: <high-DRD2-affinity hit>  → reward 0.05
  step 15: TERMINATE                 → reward 8.7   (composite oracle)
```

## Architecture

```
DrugDiscoveryEnvironment (FastAPI on HF Space)
├── reset(episode_id, difficulty)
├── step(episode_id, action)         ← state persists per episode_id
└── oracles/  → TDC binding (DRD2 default) + RDKit QED + SA + TDC CYP3A4
```

Action space: `ADD_FRAGMENT | REMOVE_FRAGMENT | SUBSTITUTE_ATOM | TERMINATE`.

Curriculum: 3 RLVE-compliant tiers — trivial (QED only, 5-fragment vocab) → easy (QED + binding, 15 fragments) → hard (4-component composite, 50 fragments, single-atom start).

## Quick reproduce

```bash
git clone https://github.com/AnshumanAtrey/pharmarl
cd pharmarl

# Validate stack (Gate 1)
pip install -e .
python scripts/validate_stack.py

# Run server locally
uvicorn server.app:app --port 8000

# Smoke test the HTTP loop without an LLM
python scripts/smoke_notebook_locally.py
```

Multi-step episode state is keyed by `episode_id` — pass the same `episode_id` to `/reset` and every `/step` in a rollout. Different rollouts (e.g. GRPO group members) use different `episode_id`s and proceed concurrently.

## Themes hit

- **Theme 3.1 Professional Tasks** — a scientific workflow loop (SELFIES editing → oracle scoring → reward-shaped optimization) directly mapping to the medicinal-chemistry hit-finding pipeline.
- **RLVE compliance** — adaptive 3-tier curriculum, procedurally seeded scaffolds, algorithmic reward verification (TDC oracles, not LLM judges).

## Held-out generalization test

We train on **DRD2 + GSK3B** (target rotated per training step) and reserve **JNK3** for evaluation — a kinase the agent never sees during training. Untrained Qwen vs trained Qwen on JNK3 measures whether the learned medicinal-chemistry primitives (basic-amine + aromatic scaffolds) transfer to an unseen kinase target. Most molRL papers don't run this comparison.

| Metric | Untrained baseline | After training |
|--------|-------------------|----------------|
| Mean cumulative reward on JNK3 | TBD | TBD |
| Delta (transfer signal) | — | TBD |

We report whatever the data shows. A null result is a null result — null > overclaim.

## Reward improvement (preview)

| Stage | Steps | Avg reward (random baseline) | Avg reward (trained) |
|-------|-------|------------------------------|----------------------|
| Trivial | 0-100 | ~+5 | TBD after training |
| Easy | 100-300 | TBD | TBD |
| Hard | 300-500 | TBD | TBD |

W&B link above shows live curves once training has run.

## Citations

- TDC: Huang et al., *Nature Chemical Biology* (2022) — `https://tdcommons.ai/`
- SELFIES: Krenn et al., *Machine Learning: Science and Technology* (2020)
- DRD2 benchmark: Olivecrona et al., *Journal of Cheminformatics* (REINVENT, 2017); also MOSES (Polykovskiy et al., 2018), GuacaMol (Brown et al., 2019)
- GRPO: Shao et al., *DeepSeekMath* (2024)
- OpenEnv: Meta PyTorch Foundation (2026)

## License

BSD-style — see LICENSE.
