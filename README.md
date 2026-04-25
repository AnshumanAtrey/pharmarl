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
| Code repository | https://github.com/AnshumanAtrey/pharmarl |
| HF Space (deployed env) | https://huggingface.co/spaces/anshumanatrey/pharmarl |
| Colab notebook (training) | `<TODO — Sahil paste share URL>` |
| Pitch video (90s) | `<TODO — Vijay paste YouTube URL>` |
| HF blog post | `<TODO — Vijay paste URL after publish>` |
| W&B training run | `<TODO — Sahil paste URL>` |
| Trained model on HF Hub | `<TODO — Sahil paste URL after push>` |

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

## How the reward is computed (and why you don't have to trust us)

The composite oracle isn't graded by us. It's graded by [Therapeutics Data Commons (TDC)](https://tdcommons.ai/) — frozen, peer-reviewed classifiers (Huang et al., *Nature Chemical Biology* 2022). The exact same DRD2 / GSK3B / JNK3 weights are used in:

- **REINVENT** (Olivecrona et al., 2017) — *Journal of Cheminformatics*
- **MolDQN** (Zhou et al., 2019)
- **GraphAF** (Shi et al., 2020) — *ICLR*
- **GFlowNets** (Bengio et al., 2021) — *NeurIPS*
- **MOSES** (Polykovskiy et al., 2018), **GuacaMol** (Brown et al., 2019)

Reviewers do not have to trust our code. `pip install pytdc` and you can reproduce every reward number we publish — the env is a thin wrapper, not a reimplementation. Run `python scripts/verify_reward_externally.py` to confirm: it scores aspirin and haloperidol against TDC directly, with **zero PharmaRL code in the loop**, and you'll see the same values we do (aspirin → ~0.0003 on DRD2, haloperidol → ~0.99). External instrument, not author-as-judge.

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
- **Patronus AI sub-theme** (consumer workflows with schema drift) — see *Novel mechanics* below.
- **Halluminate sub-theme** (multi-actor environments) — see *Novel mechanics* below.
- **Fleet AI sub-theme** (oversight agents monitoring other AI agents) — see *Novel mechanics* below.

## Novel mechanics (opt-in, flag-gated)

Both default OFF — the headline training run uses static reward against a single composite. Each mechanic has a dedicated demo cell in `colab/train_pharmarl.ipynb` and Q&A defense in `docs/qa-defense.md`.

### Mid-episode schema drift (Patronus AI sub-theme)
Real medicinal-chemistry projects have shifting constraints — early-stage potency push uncovers an ADMET liability, synthesizability tightens before scale-up. We model this directly: enable `CurriculumConfig.schema_drift_enabled` and the reward weights flip mid-episode, with the agent receiving a `drift_warning` in observation when constraints change. To our knowledge, no prior molecular RL env has dynamic reward weights.

### Multi-actor critic (Halluminate sub-theme)
Enable `CurriculumConfig.critic_enabled` and a separate logical agent — a deterministic rules-based medicinal chemist — examines each post-edit molecule and emits structured feedback (PAINS substructures, MW/LogP warnings, reactive group flags). The critique is appended to the next observation's metadata; the policy can integrate or ignore it. Rules-based not LLM-based: deterministic, ms latency, no API cost. The env contract has a clean seam — a frozen LLM critic could be swapped in here as future work.

### Pure-LLM oversight (Fleet AI sub-theme)
Enable `CurriculumConfig.oversight_enabled` and at episode end a separate LLM — Gemini 2.5 Flash by default — examines the full action trajectory and emits a structured report: `strategy_summary`, `risk_flags`, `risk_level` (low/medium/high), `explanation`. **Backward-looking** counterpart to the critic. ONE LLM call per episode, only at TERMINATE — no API quota burn during training. The env supports OpenRouter as an alternate provider. Set `GEMINI_API_KEY` (or `OPENROUTER_API_KEY`) and pass `oversight_enabled=True` in the config to demo. See `colab/train_pharmarl.ipynb` cell 14.

## Baseline spectrum — what we measured before training

We probed 6 policies on the same eval (9 episodes per target × 3 targets):

| Source | DRD2 | GSK3B | JNK3 | Mean | Cost |
|---|---|---|---|---|---|
| Random uniform | +2.78 | +2.33 | +1.78 | +2.30 | $0 |
| Scripted (4-step) | +2.90 | +3.04 | +2.50 | +2.81 | $0 |
| Llama 3.2 3B | +1.80 | +1.99 | +1.22 | +1.67 | $0.001 |
| Gemini 2.5 Flash | +2.18 | +1.10 | +2.15 | +1.81 | $0.026 |
| Llama 3.1 8B | +2.52 | +2.57 | +2.27 | **+2.45** | $0.001 |
| Llama 3.3 70B | +1.65 | +0.79 | +1.14 | +1.19 | $0.007 |
| Gemini 2.5 Pro | +4.74 | +3.40 | +2.91 | +3.68 | $0.123 |

**Inverted scaling**: 70B (+1.19) < 8B (+2.45) > 3B (+1.67), and 70B falls *below* random uniform. 70B's failure mode: it tries multi-fragment over-substituted molecules in one turn that fail Lipinski + the env's chemistry validator (parse rate also slips to 97%). In this constrained action space, raw model capacity is anti-correlated with performance past a sweet spot — *discipline beats capacity*. This is the empirical proof that the env's reward function isn't trivially gameable by raw scale.

Random and scripted policies beat 3 of the 4 LLMs we tested. Only Gemini 2.5 Pro clears the scripted baseline cleanly. Total probe spend: **$0.158**. Full table + reproducing instructions in `docs/baselines.md`.

## How we hardened the reward (not the prompt)

An RL env whose only defense against gaming is the *agent's prompt* is an env whose reward is wrong. Before any training run, we red-teamed the reward function itself.

**One real exploit caught.** `RDKit.MolFromSmiles("")` returns a 0-atom molecule (not `None`), and `QED.qed()` of nothing returns ~0.34 — so an agent submitting empty output would have farmed composite ~0.44. **Fix:** 4 lines added to `server/oracles/*.py` rejecting `mol.GetNumAtoms() == 0`, plus a regression test (`tests/test_reward_redteam.py::test_empty_string_does_not_crash`) that asserts empty → 0.0. The reward function changed; the agent prompt did not.

Three stacked structural defenses, all encoded in the reward — not the prompt:

1. **Composite oracle** — any single component getting gamed gets diluted by the other three (binding 0.40 / QED 0.25 / SA 0.15 / 1-tox 0.20).
2. **Lipinski terminal gate** — composite × 0.5 if the final molecule fails Rule of 5.
3. **Anti-degenerate guards** — zero-atom Mol → 0.0; parse failure → -0.5; cannot terminate on step 1.

14 redteam tests pin this surface: `tests/test_reward_redteam.py` covers empty SMILES, polyaromatic blobs, single carbon, action repetition, disconnected fragments, charged species, the exact failure-mode molecules Llama 70B produced, and PAINS-pattern detection. **Empirical proof:** when frontier-class Llama 70B was given this env, it tried capacity-greedy strategies and scored *worse than random uniform*. The reward signal isn't just hard to game in theory — we have data.

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
