# A10G GRPO Run — `69ed99a9d2c8bd8662bcf2ef`

- **Job:** https://huggingface.co/jobs/anshumanatrey/69ed99a9d2c8bd8662bcf2ef
- **Started:** 2026-04-26T04:45:28Z
- **Finished:** _TBD — bundle to be populated by `_fetch.py` after training completes_
- **Accelerator:** GPU
- **GPU:** NVIDIA A10G (large)
- **Model:** `unsloth/Llama-3.2-3B-Instruct`
- **GRPO steps:** 200 | G=8 | SFT warmup=0 (chat-template SYSTEM gives parse=100% without warmup)
- **Trained LoRA:** [`anshumanatrey/pharmarl-llama-3b-trained-anshuman`](https://huggingface.co/anshumanatrey/pharmarl-llama-3b-trained-anshuman)

## Stack

- `transformers==4.46.3`
- `trl==0.13.0`
- `unsloth==2025.2.15`
- `bitsandbytes==0.45.5`
- `xformers==0.0.27.post2`
- `peft>=0.12,<0.14`
- `accelerate>=0.34,<1.0`

## Live metrics (during training)

- W&B run: _TBD — will be populated when training finishes (find `View run` line in HF Job tail logs)_

## How to populate this bundle after training finishes

1. Get the W&B run path from the HF Job logs:
   ```
   wandb: 🚀 View run h200-... at: https://wandb.ai/<ENTITY>/<PROJECT>/runs/<ID>
   ```
2. Edit `_fetch.py` and set `WANDB_RUN_PATH = "<ENTITY>/<PROJECT>/runs/<ID>"`
3. Run from the repo root:
   ```bash
   .venv/bin/python runs/anshuman-a10g-llama3b/_fetch.py
   ```
4. The script downloads:
   - All W&B history (per-step metrics) → `wandb/run_history.csv` + `.json`
   - W&B summary, config, system metrics → `wandb/`
   - HF Hub model repo log files → `hf_hub/`
   - 5 standardized plots → `plots/01-…` through `05-…`
5. Commit + push so teammates can read offline.

## Audit trail (in this repo, after `_fetch.py` runs)

- `hf_hub/entry.log` — host setup + sanity checks
- `hf_hub/train.log` — full trainer stdout (per-step rewards, parse rate, W&B URL)
- `hf_hub/env_server.log` — env sidecar log
- `hf_hub/run_metadata.json` — machine-readable run metadata
- `wandb/run_history.csv` — every logged step
- `wandb/run_summary.json` — final-step values
- `plots/01-reward-mean-and-max-per-grpo-step.png`
- `plots/02-policy-loss-and-kl-divergence.png`
- `plots/03-parse-rate.png`
- `plots/04-oracle-components-qed-docking-sa-toxicity.png`
- `plots/05-grpo-total-loss.png`

## Comparison with `runs/vijay-h200-llama1b/`

Vijay's H200 run with Llama-3.2-1B-Instruct hit `parse_rate=0%` from step 1 — the 1B model never emitted parseable JSON, so all rollouts used the fallback action and GRPO had no real signal. The trained LoRA there is effectively untrained. See [`runs/vijay-h200-llama1b/plots/03-parse-rate-stuck-at-zero.png`](../vijay-h200-llama1b/plots/03-parse-rate-stuck-at-zero.png).

This A10G run uses Llama-3.2-3B-Instruct with the strong chat-template SYSTEM prompt added in commit `52808d8`. Per the live HF Job logs at step 189 of 200, **`parse_rate=100%` throughout** and `mean_reward` is meaningful (~+1.6-2.0 on easy tier with real molecules like `CCCCC(C)C1CN(Cc2ccccc2)CC(=O)C1C(=O)O` at QED 0.78).

The two runs together form a clean A/B: 1B-without-SFT (null) vs 3B-without-SFT-but-better-prompt (real training).
