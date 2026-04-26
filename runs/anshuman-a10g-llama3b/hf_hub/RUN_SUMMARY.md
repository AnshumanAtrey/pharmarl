# A10G GRPO Run — `69ed99a9d2c8bd8662bcf2ef`

- **Job:** https://huggingface.co/jobs/anshumanatrey/69ed99a9d2c8bd8662bcf2ef
- **Started:** 2026-04-26T04:45:28Z
- **Finished:** 2026-04-26T10:52:00Z (approx, last step at ETA=0min)
- **Wall time:** ~6 hours
- **Trainer exit code:** 0
- **Accelerator:** GPU
- **GPU:** NVIDIA A10G (large)
- **Model:** `unsloth/Llama-3.2-3B-Instruct`
- **GRPO steps:** 200 | G=8 | SFT warmup=0
- **Trained LoRA:** [`anshumanatrey/pharmarl-llama-3b-trained-anshuman`](https://huggingface.co/anshumanatrey/pharmarl-llama-3b-trained-anshuman)
- **W&B run:** [atrey-dev/pharmarl/runs/hg0rkgyr](https://wandb.ai/atrey-dev/pharmarl/runs/hg0rkgyr)

## Stack

- `transformers==4.46.3`
- `trl==0.13.0`
- `unsloth==2025.2.15`
- `bitsandbytes==0.45.5`
- `xformers==0.0.27.post2`
- `peft==0.13.2`
- `accelerate==0.34.2`

## Headline numbers

| Metric | Value |
|---|---|
| Final mean_reward (step 199, easy tier) | **+2.079** |
| Peak max_reward (step 96) | **+8.842** |
| `verifier/parse_rate` | **100% throughout** (vs Vijay's 0%) |
| Trivial tier (steps 0-99) | mean_reward ~+4-5, peak +8.5 |
| Easy tier (steps 100-199) | mean_reward ~+1.5-2, peak +5.9 |
| Best molecule found (step 150) | `CC1C(C(=O)O)C(c2ccccc2)CCN1Cc1ccncc1` (QED 0.94, docking 0.23) |

## Best molecules per checkpoint (audit trail)

| Step | SMILES | Reward | QED | Docking |
|---|---|---|---|---|
| 0 | `O=C(O)CCc1c(C(=O)O)cc2ccccc2c1C(=O)O` | +7.525 | 0.78 | 0.006 |
| 25 | `O=C(O)c1ccncc1C(=O)O` | +5.982 | 0.67 | 0.005 |
| 50 | `CCOC(=O)c1ccc(C(=O)O)cn1` | +6.865 | 0.73 | 0.000 |
| 75 | `CCC(C)CC(C)CC(O)CC(C(=O)O)c1ccccc1` | +7.711 | 0.72 | 0.000 |
| 100 | `CCCCCCCCCCCCCNc1nccc(C(=O)O)n1` | +2.494 | 0.48 | 0.010 |
| 125 | `CCCCCCCCCCCCCCNc1nccc(C(=O)O)n1` | +1.562 | 0.43 | 0.014 |
| **150** | **`CC1C(C(=O)O)C(c2ccccc2)CCN1Cc1ccncc1`** | **+4.132** | **0.94** | **0.229** |
| 175 | `CCCCC(C)C1CN(Cc2ccccc2)CC(=O)C1C(=O)O` | +2.886 | 0.78 | 0.003 |

## Comparison with `vijay-h200-llama1b`

| Aspect | Vijay's H200 (Llama-1B) | This run (A10G, Llama-3B) |
|---|---|---|
| `parse_rate` | 0% throughout (model never produced JSON) | **100% throughout** |
| Final mean_reward | +0.809 (degraded from +5.0) | +2.079 (easy tier) |
| Best peak reward | ~+1.4 | **+8.842** |
| Real molecule generation | No (used fallback action) | Yes — diverse, drug-like |
| Effectively trained? | **No** (gradient signal was zero) | **Yes** |

The 3B model + chat-template SYSTEM prompt added in commit `52808d8` was the difference. 1B never parsed JSON; 3B parsed 100%.

## Audit trail (in this repo)

- `wandb/run_history.csv` — every logged step (208 rows × 30 cols)
- `wandb/run_summary.json` — final-step values (27 keys)
- `wandb/run_config.json` — trainer config snapshot
- `wandb/run_meta.json` — W&B run metadata + URL
- `wandb/system_metrics.csv` — GPU/CPU time series (3010 rows)
- `plots/01-reward-mean-and-max-per-grpo-step.png`
- `plots/02-policy-loss-and-kl-divergence.png`
- `plots/03-parse-rate.png` (flat 100% — opposite of Vijay's plot)
- `plots/04-oracle-components-qed-docking-sa-toxicity.png`
- `plots/05-grpo-total-loss.png`
