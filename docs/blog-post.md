# PharmaRL: An OpenEnv-Native Molecular RL Environment Where Discipline Beats Capacity

*Built for the Meta PyTorch OpenEnv Hackathon (Apr '26) by AI Mafias — Anshuman, Sahil, Vijay.*

## TL;DR

We built the first OpenEnv-native molecular environment where an LLM is the policy. Same canonical molecular RL benchmark every paper uses (DRD2 binding from Therapeutics Data Commons), new policy class. Two opt-in mechanics — mid-episode schema drift and a multi-actor medicinal-chemist critic — give us coverage on Patronus AI and Halluminate sub-theme bonuses. Most surprising finding: **Llama 70B scores worse than random uniform on this env.** Capacity isn't the bottleneck; discipline is.

- 🤗 **Live env**: https://huggingface.co/spaces/anshumanatrey/pharmarl
- 💻 **Code**: https://github.com/AnshumanAtrey/pharmarl
- 📔 **Colab**: `<TODO>`
- 🎬 **90s pitch**: `<TODO>`

## The gap

Every modern molecular RL paper — REINVENT, MolDQN, GraphAF, GFlowNets — uses a GNN or RNN policy trained from scratch. That's useful for research, but the open-source LLM ecosystem has been moving in a different direction: structured-action chat agents you can drop into any environment via JSON. There's no standard environment for training a chat-style LLM as the policy on a real molecular task. PharmaRL fills that gap.

The agent issues structured JSON actions — `ADD_FRAGMENT`, `REMOVE_FRAGMENT`, `SUBSTITUTE_ATOM`, `TERMINATE` — over a 10-20 step episode. SELFIES (Krenn et al., 2020) guarantees every output is chemically valid. Reward is a composite of TDC's binding classifier + RDKit's QED + synthetic accessibility + CYP3A4 toxicity, gated by Lipinski's Rule of 5.

## Architecture

```
DrugDiscoveryEnvironment (FastAPI on HF Space)
├── reset(episode_id, difficulty)
├── step(episode_id, action)         ← state persists per episode_id
└── oracles/  → TDC binding (DRD2 default) + RDKit QED + SA + TDC CYP3A4
```

Episode state is keyed by `episode_id` so a GRPO group of 8 concurrent rollouts each gets isolated state. Curriculum has three RLVE-compliant tiers — trivial (QED only) → easy (QED + binding) → hard (full 4-component composite).

## Reward design — designed to NOT be gamed

A reward function people will actually train against has to be hard to game. We layered three independent defenses:

1. **Composite oracle** — any single component getting gamed gets diluted by the other three.
2. **Lipinski gate** — terminal reward halved if the final molecule fails Rule of 5. Eliminates the "build the biggest molecule possible" exploit.
3. **Anti-degenerate guards** — zero-atom Mol → 0.0; parse failure → -0.5; cannot terminate on step 1.

We pinned this with 14 adversarial reward tests in `tests/test_reward_redteam.py` covering empty SMILES, polyaromatic blobs, action repetition, disconnected fragments, charged species, and the exact failure-mode molecules our biggest baseline LLM produced.

## Inverted scaling — the surprising finding

Before training, we ran six off-the-shelf policies on the same eval (9 episodes per target × 3 targets, easy difficulty):

| Policy | Mean cumulative reward | Cost |
|---|---|---|
| Random uniform | +2.30 | $0 |
| Scripted 4-step | +2.81 | $0 |
| Llama 3.2 3B Instruct | +1.67 | $0.001 |
| Gemini 2.5 Flash | +1.81 | $0.026 |
| **Llama 3.1 8B Instruct** | **+2.45 ← sweet spot** | $0.001 |
| Llama 3.3 70B Instruct | +1.19 | $0.007 |
| Gemini 2.5 Pro | +3.68 | $0.123 |

Across the Llama family, **8B beats both 3B and 70B**. 70B specifically falls *below random uniform* — it tries to design oversized multi-fragment molecules in single turns and runs head-first into the env's Lipinski + chemistry validators. Parse rate also drops to 97%.

This isn't just a fun graph. It's empirical proof that the env's reward design penalizes capacity-greedy strategies, which is exactly what you want from an RL environment intended for actual training. **Random and scripted policies beat 3 of the 4 LLMs.** Only Gemini 2.5 Pro clears the scripted baseline cleanly. Total probe spend: $0.158.

The thesis behind training a 1.5B Qwen with GRPO on this env follows directly: targeted reward optimization can outperform raw model capacity.

## Headline result — trained Qwen 1.5B

We trained Qwen 2.5 1.5B with Unsloth + TRL's GRPO across `[N]` steps on two targets (DRD2 + GSK3B), with JNK3 held out for transfer evaluation.

**Training curve**: `<EMBED docs/plots/training_curve.png>`

**Per-target comparison vs baselines**: `<EMBED docs/plots/per_target_comparison.png>`

**Distribution per target**: `<EMBED docs/plots/distribution_box.png>`

`[FILL] Trained Qwen mean cumulative: +X.XX` — comparing against the baseline table above:

- Beats random uniform (+2.30)? **`[YES/NO]`**
- Beats scripted heuristic (+2.81) — *the "agent learned" floor*? **`[YES/NO]`**
- Beats untrained Llama 8B sweet spot (+2.45)? **`[YES/NO]`**
- Approaches Gemini 2.5 Pro (+3.68)? **`[YES/NO]`**

`[FILL]` based on the actual JSON output of `colab/train_pharmarl.ipynb` cell 13.

## Held-out target — JNK3 generalization test

Most molRL papers don't run a held-out target measurement at all. We do:

| | Untrained Qwen on JNK3 | Trained Qwen on JNK3 |
|---|---|---|
| Mean cumulative | `[BEFORE]` | `[AFTER]` |
| Std | `[BEFORE_STD]` | `[AFTER_STD]` |
| Delta | — | **`[DELTA]`** |

**Honest scope**: GSK3B and JNK3 are both serine/threonine kinases — this is *intra-family* transfer to a novel kinase, not cross-family. We're not claiming the agent learned binding principles in general. We're showing whether kinase-trained skill transfers to an unseen kinase. If the result is null, we report it as null — null is still a result.

## Two opt-in mechanics for sub-theme bonuses

Both default OFF; they each have a dedicated demo cell in the Colab notebook.

### Mid-episode schema drift (Patronus AI sub-theme)

Real medicinal-chemistry projects don't have static optimization criteria. A potency push uncovers an ADMET liability; the synthesizability bar tightens before scale-up. The constraints shift mid-development.

We model this directly. Enable `CurriculumConfig.schema_drift_enabled` and the reward weights flip mid-episode according to a sampled drift profile (`static`, `early_admet`, `late_potency`). The agent sees a `drift_warning` in the observation when constraints change.

The Patronus AI sub-theme rewards exactly this kind of consumer-workflow-with-changing-rules environment. To our knowledge, no prior molecular RL env has dynamic reward weights.

### Multi-actor medicinal chemist critic (Halluminate sub-theme)

Enable `CurriculumConfig.critic_enabled` and a separate logical agent — a deterministic rules-based medicinal chemist — examines each post-edit molecule and emits structured feedback: PAINS substructures (thiocarbonyl, rhodanine, nitroaromatic Michael acceptors), Lipinski-flavored property warnings, reactive group flags (alkyl halide, epoxide, anhydride). The critique is appended to the next observation's metadata under `critique`; the policy can integrate the feedback (e.g., REMOVE_FRAGMENT to clear a flagged group) or ignore it.

**Why rules-based, not an LLM critic?** Two reasons. **Deterministic**: same molecule always gets the same critique, so critic-conditioned training is reproducible. An LLM critic introduces stochasticity that interferes with reward signal cleanliness. **Fast**: ms latency vs ~1s per LLM call. The env contract has a clean seam — a frozen LLM critic can be swapped in here as future work.

The Halluminate sub-theme rewards multi-actor environments where the agent interacts with separate logical agents to discover and achieve the task. Whether the actor is itself an LLM or a deterministic rules engine is implementation detail; what matters is the structured-feedback channel.

## What's next

This is hackathon scope (built in 18 hours, three teammates, one subscription tier of compute). The natural follow-ups, in order:

1. **More training** — 200 GRPO steps is the lower bound for showing a curve on T4; 2000+ steps with better hyperparameters likely closes the gap to Gemini Pro.
2. **Frozen LLM critic** — swap the rules-based critic for a frozen Qwen-7B with a chemist persona; richer feedback at the cost of latency.
3. **Cross-family held-out** — find a non-kinase, non-QED-correlated TDC oracle for a stronger transfer claim. We investigated several candidates; documented which work and don't in `docs/cross_family_attempt.md`.
4. **Stage 2 docking deployment** — the env supports pyscreener-backed docking against NSP15 / EGFR / ABL / BACE1. The chemistry stack doesn't pip-install cleanly in stock HF Spaces; a dedicated Docker build with OpenBabel + AutoDock Vina is the path forward.

## Why this matters for OpenEnv

OpenEnv's pitch is environments as portable, versioned software artifacts. PharmaRL is a small but pointed test of that thesis: a real chemistry environment with verifiable reward, deployed as a Hugging Face Space, runnable from a Colab notebook by any team — and a *demonstration* that off-the-shelf frontier LLMs alone don't solve it. The path from "we have a good env" to "we have a useful trained model" requires both — and the hackathon tooling stack (Unsloth + TRL + OpenEnv + HF Spaces) made the round-trip practical in under a day.

Try it: https://huggingface.co/spaces/anshumanatrey/pharmarl

---

*PharmaRL is open source under BSD. Code at https://github.com/AnshumanAtrey/pharmarl. Built by AI Mafias for the Meta PyTorch OpenEnv Hackathon, April 2026.*
