# PharmaRL

**Multi-step molecular drug discovery environment for OpenEnv.** An LLM agent iteratively edits SMILES/SELFIES molecular strings to design drug-like ligands. Reward is computed by **Therapeutics Data Commons (TDC)** oracles plus **RDKit** validity checks — *no LLM judge in the reward path* (RLVR, guide §11/Q4).

> Built for **Meta PyTorch OpenEnv Hackathon Apr '26** by **AI Mafias** (Anshuman, Sahil, Vijay).

The agent: a **Llama-3.2-1B-Instruct** model post-trained with **SFT format priming → GRPO** (multi-turn, group-normalized advantage, PPO-clip surrogate, Schulman-K3 KL against a frozen reference). The reward: a **4-pillar composite** (binding · drug-likeness · synthesizability · toxicity) with a Lipinski Rule-of-5 gate to prevent reward hacking.

The **scientific story**: SARS-CoV-2 main protease (Mpro) is a validated antiviral target — Pfizer's Paxlovid (nirmatrelvir) hits exactly this protein. Generative drug design against Mpro is an active research area; we train an open-weight 1B-parameter model to do it in <2h on a free Colab T4.

---

## Submission URLs

| Resource | URL |
|----------|-----|
| HF Space (deployed env) | `<TODO: https://huggingface.co/spaces/anshumanatrey/pharmarl>` |
| Colab notebook (training) | `<TODO: colab share URL>` |
| Code repository | `<TODO: https://github.com/anshumanatrey/pharmarl>` |
| Pitch video (90s) | `<TODO: YouTube or HF blog URL>` |
| W&B training run | `<TODO: wandb.ai run URL>` |
| Trained model on HF Hub | `<TODO: https://huggingface.co/anshumanatrey/pharmarl-llama-trained>` |

---

## What this is

A **Type B** OpenEnv environment (real RL with state dynamics, not text grader). The agent edits a molecule across 10–20 sequential steps; each edit changes what's possible in the next step. Reward is sparse (terminal composite oracle score) plus dense shaping (Lipinski compliance per step, invalid-action penalty).

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
├── reset()     → procedurally-sampled starting molecule (36-seed pool)
├── step()      → mutates SELFIES, computes reward, returns observation + per-component breakdown
└── oracles/    → DRD2 classifier (default, MOSES/GuacaMol benchmark)
                  + RDKit QED + RDKit/TDC SA + TDC CYP3A4 toxicity
                  (opt-in: pyscreener docking against NSP15/EGFR/ABL/BACE1
                   with PHARMARL_ENABLE_DOCKING=1 + Vina + OpenBabel)
```

**Action space**: `ADD_FRAGMENT | REMOVE_FRAGMENT | SUBSTITUTE_ATOM | TERMINATE` over a difficulty-scaled fragment vocabulary (5 / 15 / 50+ SMILES).

**Curriculum**: 3-tier (trivial → easy → hard) with **adaptive RLVE promotion** based on rolling success rate (guide §22, Q35). Step-threshold fallback if `USE_ADAPTIVE_CURRICULUM=False`.

**Reward composition** (terminal):
```
composite = 0.40 · docking + 0.25 · QED + 0.15 · SA + 0.20 · (1 − toxicity)
final     = composite × 10  ×  (0.5 if Lipinski fails else 1.0)
```

## Why a chemist or judge should trust this

- **Verifiable rewards** (RLVR / Q21): TDC + RDKit deterministic functions, no LLM judges
- **Reward red-team test** (`tests/test_reward_redteam.py`, guide Q57) asserts a real drug beats trivial reward-gamers and that the Lipinski gate halves composite for non-drug-like molecules
- **Multi-pillar composition** (guide §7): single-axis hacking can't win — small QED-rich molecules without binding still lose to balanced drugs
- **SELFIES validity guarantee** (Krenn 2020): every action produces a chemically valid molecule
- **DRD2 default oracle**: the canonical MOSES/GuacaMol benchmark used in MolDQN, REINVENT, GraphAF, GFlowNet — instantly recognizable

## What is *not* claimed

- We do **not** use a Dengue-specific oracle (none exists in TDC's free oracle catalog)
- We do **not** rotate the docking target per `reset()` — target is fixed per session (procedural variation is on the *seed molecule*, sampled from a 36-molecule pool per tier)
- We use Llama-3.2-1B for the open-weight story; swap in the notebook if you prefer Qwen2.5-1.5B

These are honest limitations. We list them upfront because the guide (Q33, Q57) flags overclaiming as a credibility risk.

## Quick reproduce

```bash
git clone <repo>
cd pharmarl
pip install -e .

# 1. Validate the science stack works (Phase 0 gate)
python scripts/validate_stack.py

# 2. Run the env locally
uvicorn server.app:app --port 8000

# 3. Run the reward red-team (verifies no obvious hacks)
pytest tests/test_reward_redteam.py -v

# 4. Run the §19-format demo (no LLM needed — random vs scripted policy)
python -m examples.demo --base-url http://localhost:8000 \
    --episodes 10 --policies random scripted

# 5. Train (primary path: Colab T4):
#    Open colab/train_pharmarl.ipynb, set PHARMARL_ENV_URL to your HF Space, run all cells.
#    The notebook does SFT format priming → GRPO automatically.

# 5b. Or train via the standalone script (Path C fallback if Colab dies):
python -m scripts.train_grpo \
    --env-url https://YOUR-USER-pharmarl.hf.space \
    --max-steps 200 \
    --hf-repo YOUR-USER/pharmarl-llama-trained \
    --hf-token "$HF_TOKEN"
# See docs/training-fallback.md for HF Spaces GPU / Modal / RunPod runbooks.
```

## Themes hit

- **Theme 3.1** Professional Tasks — "scientific workflow loops (papers → code → experiments)" (verbatim in judging doc)
- **Theme 4** Self-Improvement — RLVE adaptive difficulty (3 tiers, success-rate-driven promotion, procedural seeds)
- **Sub-theme bonus**: Snorkel AI (composite oracle as simulated experts-in-the-loop)

## Reward improvement

| Stage | Steps | Avg reward (random policy) | Avg reward (trained Llama) |
|-------|-------|----------------------------|----------------------------|
| Trivial (QED only) | 0–100 | TBD after demo run | TBD after training |
| Easy (QED + binding) | 100–300 | TBD | TBD |
| Hard (full 4-pillar) | 300–500 | TBD | TBD |

W&B link above shows live curves with **per-component breakdowns** (binding / QED / SA / toxicity logged separately so you can see *which axis* improved).

## Citations

- TDC: Huang et al., *Nature Chemical Biology* (2022) — `https://tdcommons.ai/`
- SELFIES: Krenn et al., *Machine Learning: Science and Technology* (2020)
- GRPO: Shao et al., *DeepSeekMath* (2024)
- OpenEnv: Meta PyTorch Foundation (2026)
- DRD2 benchmark: Olivecrona et al., *J. Cheminformatics* (2017) — REINVENT

## License

BSD-style — see LICENSE.
