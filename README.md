# PharmaRL

Multi-step molecular drug discovery environment for OpenEnv. An LLM agent iteratively edits SELFIES molecular strings to design drug-like ligands. Reward is computed by Therapeutics Data Commons (TDC) oracles plus RDKit validity checks.

**Stage 1 (default)**: scores against DRD2 — the canonical MOSES/GuacaMol molecular RL benchmark used in every paper from MolDQN through REINVENT. Trains in minutes on Colab T4.

**Stage 2 (opt-in)**: clinical docking targets — SARS-CoV-2 NSP15 (`7l11`), EGFR T790M (`2rgp`), ABL kinase (`1iep`), β-secretase 1 (`4rlu`) — via TDC's pyscreener-backed docking suite. Enable with `PHARMARL_ENABLE_DOCKING=1` and OpenBabel + AutoDock Vina installed.

> Built for Meta PyTorch OpenEnv Hackathon Apr '26 by **AI Mafias** (Anshuman, Sahil, Vijay).

---

## Submission URLs

| Resource | URL |
|----------|-----|
| HF Space (deployed env) | `<TODO: https://huggingface.co/spaces/anshumanatrey/pharmarl>` |
| Colab notebook (training) | `<TODO: colab share URL>` |
| Code repository | `<TODO: https://github.com/anshumanatrey/pharmarl>` |
| Pitch video (90s) | `<TODO: YouTube or HF blog URL>` |
| W&B training run | `<TODO: wandb.ai run URL>` |
| Trained model on HF Hub | `<TODO: https://huggingface.co/anshumanatrey/pharmarl-qwen-trained>` |

---

## What this is

A **Type B** OpenEnv environment (real RL with state dynamics, not text grader). The agent edits a molecule across 15-20 sequential steps; each edit changes what's possible in the next step. Reward is sparse (terminal composite oracle score) plus dense shaping (Lipinski compliance per step).

```
Episode trajectory (15 steps):
  step 0:  C                    → reward 0.0
  step 1:  CC                   → reward 0.05  (Lipinski OK, +shaping)
  step 5:  CC(=O)c1ccccc1       → reward 0.05
  step 14: <complex inhibitor>  → reward 0.05
  step 15: TERMINATE            → reward 8.7   (composite oracle)
```

## Architecture

```
DrugDiscoveryGym (FastAPI on HF Space)
├── reset()     → starting molecule by difficulty tier
├── step()      → mutates SELFIES, computes reward, returns observation
└── oracles/    → TDC Mpro docking + RDKit QED + SA + TDC CYP toxicity
```

Action space: `ADD_FRAGMENT | REMOVE_FRAGMENT | SUBSTITUTE_ATOM | TERMINATE`.

Curriculum: trivial → easy → hard (RLVE — adaptive difficulty).

## Quick reproduce

```bash
git clone https://github.com/anshumanatrey/pharmarl
cd pharmarl

# Validate stack (Gate 1)
pip install -e .
python scripts/validate_stack.py

# Run server locally
uvicorn server.app:app --port 8000

# Run baseline agent
python inference.py --base-url http://localhost:8000 --difficulty trivial
```

## Themes hit

- **Theme 3.1** Professional Tasks — "scientific workflow loops (papers → code → experiments)" (verbatim in judging doc)
- **Theme 4** Self-Improvement — RLVE adaptive curriculum (3 difficulty tiers, procedural seeds)
- **Sub-theme bonus**: Snorkel AI (simulated experts-in-the-loop via composite oracle)

## Reward improvement (preview)

| Stage | Steps | Avg reward (random) | Avg reward (trained) |
|-------|-------|---------------------|----------------------|
| Trivial | 0-100 | -0.2 | TBD after training |
| Easy | 100-300 | -1.1 | TBD |
| Hard | 300-500 | -2.4 | TBD |

W&B link above shows live curves.

## Citations

- TDC: Huang et al., *Nature Chemical Biology* (2022) — `https://tdcommons.ai/`
- SELFIES: Krenn et al., *Machine Learning: Science and Technology* (2020)
- GRPO: Shao et al., *DeepSeekMath* (2024)
- OpenEnv: Meta PyTorch Foundation (2026)

## License

BSD-style — see LICENSE.
