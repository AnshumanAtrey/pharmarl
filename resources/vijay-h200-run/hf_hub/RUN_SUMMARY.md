# H200 GRPO Run — 69edba4bd2c8bd8662bcf723

- **Job:** https://huggingface.co/jobs/?/69edba4bd2c8bd8662bcf723
- **Started:** 2026-04-26T07:11:36Z
- **Finished:** 2026-04-26T09:32:08Z
- **Wall time:** 8361.1s
- **Trainer exit code:** 0
- **Accelerator:** gpu
- **GPU:** NVIDIA H200 (capability (9, 0))
- **Model:** unsloth/Llama-3.2-1B-Instruct
- **GRPO steps:** 200 | G=8 | SFT warmup=120

## Stack

- `transformers==4.46.3`
- `trl==0.13.0`
- `unsloth==?`
- `bitsandbytes==0.45.5`
- `xformers==0.0.29.post3`
- `peft==0.19.1`
- `accelerate==1.13.0`

## Live metrics

- W&B run: https://wandb.ai/vijaykota2776-itm/pharmarl

## Audit trail (in this repo)

- [`logs/entry.log`](logs/entry.log) — host setup + sanity checks
- [`logs/train.log`](logs/train.log) — full trainer stdout (per-step rewards, parse rate, W&B URL)
- [`logs/env_server.log`](logs/env_server.log) — env sidecar log
- [`logs/run_metadata.json`](logs/run_metadata.json) — machine-readable run metadata
- Adapter: `lora_final/` (pushed by `train_grpo.py` itself)