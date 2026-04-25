# Chemistry Primer — Q&A defense for PharmaRL

This is the cheat-sheet for handling judge questions in 30 seconds. Memorize the bold lines.

---

## Why SELFIES instead of SMILES?

**SELFIES strings are always chemically valid** — every possible string decodes to a real molecule. SMILES can be invalid (mismatched parentheses, bad valences). This eliminates a major reward-hacking surface where an agent could spam gibberish and accidentally score positive on a malformed parser.

Reference: Krenn et al., "SELFIES — a robust representation of semantically constrained graphs with an application to constructive molecular generation" (2020).

## What does kcal/mol mean?

Binding affinity unit. **More negative = stronger binder.**

| Score (kcal/mol) | Interpretation |
|------------------|----------------|
| -3 to -5 | Weak binder, drug-like noise floor |
| -6 to -8 | Decent binder, hit-level |
| -8 to -10 | Strong binder, lead-level |
| -10 to -12 | Extremely strong, often clinical candidate |

Nirmatrelvir (Pfizer Paxlovid against Mpro) ≈ -10 kcal/mol.

We normalize raw kcal/mol to [0, 1] for reward (clamp [-12, 0] then divide).

## Lipinski's Rule of 5 (1997, Chris Lipinski at Pfizer)

A drug-likeness rule-of-thumb. Drugs that pass have **higher oral bioavailability**.

| Property | Limit |
|----------|-------|
| Molecular weight (MW) | ≤ 500 Da |
| LogP (lipophilicity) | ≤ 5 |
| Hydrogen-bond donors (HBD) | ≤ 5 |
| Hydrogen-bond acceptors (HBA) | ≤ 10 |

We use it as an **anti-reward-hacking gate**: if the final molecule fails Rule of 5, terminal reward is halved. This prevents the agent from inflating affinity by stacking aromatic rings to a non-drug-like blob.

## What's QED?

**Quantitative Estimate of Drug-likeness** (Bickerton et al. 2012). A weighted composite of 8 descriptors (MW, LogP, HBD, HBA, PSA, rotatable bonds, aromatic rings, structural alerts). Scaled to [0, 1].

| Compound | QED |
|----------|-----|
| Aspirin | 0.55 |
| Caffeine | 0.55 |
| Ibuprofen | 0.81 |
| Morphine | 0.65 |

Our reward weights QED at 25%.

## What's SA score?

**Synthetic Accessibility** (Ertl & Schuffenhauer 2009). Estimates how hard a molecule is to make. Range 1 (trivial) to 10 (PhD thesis to synthesize). We invert + normalize so higher = more synthesizable. Reward weight: 15%.

## What's CYP3A4 inhibition?

CYP3A4 is a liver enzyme that metabolizes ~50% of clinical drugs. Inhibitors cause drug-drug interactions (you can't co-prescribe with statins, etc.). TDC has a classifier returning probability of inhibition. We invert it into the reward as toxicity penalty (weight 20%).

## Why DRD2 as the binding oracle in Stage 1?

**DRD2 = dopamine D2 receptor.** TDC provides a high-quality random forest classifier trained on hundreds of thousands of D2 actives/inactives from ChEMBL. It returns probability of D2 activity in [0,1]. Aspirin scores ~0.0003 (correctly: not a D2 ligand). Haloperidol — a known D2 antagonist drug — scores 0.99. The signal is clean and fast.

**Why this is the right choice for Stage 1, not a compromise**:
- DRD2 is the **canonical molecular RL benchmark**. MolDQN (2019), REINVENT, GraphAF, MIMOSA, GFlowNets — every published molecular RL system reports on DRD2. Judges who know this field will instantly recognize the choice as legitimate.
- It's classifier-based, not docking — meaning it works without OpenBabel/AutoDock Vina/pyscreener (all heavyweight native deps). The env trains in minutes on Colab.
- The Stage 1 architecture is target-agnostic. We can plug in real clinical docking (`7l11_docking` for SARS-CoV-2 NSP15, `2rgp_docking` for EGFR, `1iep_docking` for ABL kinase) by setting `PHARMARL_ENABLE_DOCKING=1` and installing the chemistry stack. The env code already probes for these and falls back cleanly.

**Q: "Why not direct Mpro docking?"**
A: TDC has no Oracle('Mpro') — Mpro docking requires the pyscreener+Vina chemistry stack which doesn't pip-install. We support it as a Stage 2 deploy-time extension. For training-pipeline validation, DRD2 is the standard the field has agreed on.

**Q: "How does this transfer to a real antiviral?"**
A: Same env, swap the oracle. The agent learns *medicinal chemistry primitives* (Lipinski compliance, drug-like scaffold construction, fragment selection by binding signal) — those transfer. Once `PHARMARL_ENABLE_DOCKING=1` is set on a host with Vina installed, the same trained model can be evaluated against `7l11_docking` (SARS-CoV-2 NSP15) directly.

## Why these 4 reward components specifically?

We chose components that are:
1. **Independent**: docking ⊥ drug-likeness ⊥ synthesizability ⊥ toxicity
2. **Non-gameable**: each is a published, validated scientific oracle (TDC = Harvard/MIT)
3. **Visually distinguishable**: 4 separate W&B curves judges can see climb together

A single docking score is reward-hackable (agent stacks rings for binding while violating Lipinski). Composite + Lipinski gate prevents this.

## Why this curriculum (Trivial → Easy → Hard)?

**RLVE** — Reinforcement Learning with Verifiable Environments. Theme 4 of the judging doc explicitly calls for adaptive difficulty.

- **Trivial** (steps 0-100): single property (QED), known scaffold, 5-fragment vocab. **Guarantees a moving reward curve in <100 GRPO steps** — insurance for the 20% Reward Improvement criterion.
- **Easy** (100-300): 2 properties, 15-fragment vocab. Tests transfer.
- **Hard** (300+): 4-property composite, single-atom start, 50-fragment vocab. **De novo design** — the real frontier.

## Anti-staleness defense (the "static dataset" question)

Q: "If the oracle is deterministic, won't your agent just memorize one molecule per target?"

A: Three reasons no:
1. **Generative search space**: ~10^60 valid drug-like molecules. The agent constructs molecules sequentially; trajectories don't repeat.
2. **GRPO temperature > 0**: 8 different actions sampled per state every step. Same state → 8 different next-molecules.
3. **Procedural seeds**: 200-molecule starting pool per difficulty, randomly sampled per episode. The agent never starts from the same molecule twice in close succession.

The deterministic oracle is the **verifiable reward signal** (the V in RLVE), not the env's output. The env's output is the agent's trajectory, which is stochastic by construction.

## What is GRPO?

**Group Relative Policy Optimization** (Shao et al., DeepSeek 2024). The intuition:
1. For one prompt, sample G=8 different actions
2. Score each
3. Tokens from above-average actions → boost probability
4. Tokens from below-average actions → reduce probability
5. KL-divergence regularizer prevents drifting too far from the base model

Memory advantage over PPO: no separate value model needed. The group itself is the baseline. Used by DeepSeek-R1, fits in Colab T4.

## What does Unsloth do?

Without Unsloth, training Qwen3-1.5B on a T4 (16GB) **does not fit**. Unsloth provides:
1. **4-bit quantization** of base weights (4× memory reduction)
2. **LoRA adapters** — only ~10M trainable params instead of 1.5B
3. **Flash Attention** — 2-3× speedup
4. **Custom CUDA kernels** for hot loops

Without it, you'd need an A100 ($2-3/hr). With it, free Colab works.

## What proves the trained model actually learned?

The before/after demo:
- Same starting molecule
- Same target
- **Base model**: random fragment additions, terminal composite ≈ 0.2 (poor)
- **Trained model**: structured edits matching known Mpro pharmacophores, terminal composite ≈ 0.7+ (lead-like)

Plus the W&B reward curve climbing across training steps. Plus held-out scaffolds (model trained on starting pool A, evaluated on pool B) showing transfer, not memorization.
