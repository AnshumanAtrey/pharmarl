---
title: PharmaRL Training (H200)
emoji: 🚀
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
hardware: h200
pinned: false
license: bsd-3-clause
---

# PharmaRL training Space — H200 parallel run

**Faster-finish insurance for the deadline.** This Space runs the same GRPO
training as the Colab / a10g paths, but on a single H200 (Hopper, 141 GB
VRAM). Wall time ≈ 1.7 h, cost ≈ $5/hr, expected finish ≈ 3 h before deadline.

This is the H200 sibling to the original t4-small / a10g fallback. It exists
in parallel — Anshuman's a10g run is the primary; this is the parachute.

## What it does

On startup, runs `scripts/train_grpo.py` against the deployed env URL passed
via the `PHARMARL_ENV_URL` Space secret. SFT format priming → GRPO training →
pushes the trained LoRA to HF Hub. A minimal HTTP server stays up so the
Space doesn't show as "Crashed."

## Why a different image vs. the t4-small fallback

The headline t4 / a10g fallback uses `nvidia/cuda:12.1.1-cudnn8` + PyTorch
2.4. That stack will **not** run on H200 — sm_90 needs xformers ≥ 0.0.30 and
bitsandbytes ≥ 0.45.5, neither of which is present in the older image. This
Dockerfile pins the Hopper-compatible matrix:

| Component | Version | Why |
|---|---|---|
| Base image | `pytorch:2.6.0-cuda12.4-cudnn9-devel` | Hopper-aware CUDA + cuDNN |
| `bitsandbytes` | `>=0.45.5` | LLM.int8 on Hopper (older versions ignore sm_90) |
| `xformers` | `>=0.0.30` | sm_90 attention kernels (0.0.27 lacks them) |
| `transformers` | `==4.46.3` | tested-good with unsloth 2025.2.15 |
| `trl` | `==0.13.0` | GRPOTrainer signature this trainer expects |
| `unsloth` | `==2025.2.15` | matched against transformers 4.46.3 |

## Sanity signals (in order, as the Space starts)

1. `GPU: NVIDIA H200`, capability `(9, 0)` — confirms hardware
2. `unsloth import OK` — confirms stack is consistent
3. First training step prints `parse=100%` — confirms env reachable
4. Every 25 steps logs without error — confirms run is stable
5. Adapter pushed to `$HF_REPO` at end (e.g. `vijay2776/pharmarl-llama-3b-trained-vijay-h200`)

## How to deploy

1. **Create the Space** (or push to existing one):
   ```bash
   # From the pharmarl repo root, on the vijay-h200 branch
   huggingface-cli repo create pharmarl-training-h200 --type space --space_sdk docker
   git clone https://huggingface.co/spaces/YOUR-USER/pharmarl-training-h200 /tmp/pharmarl-h200
   cp -r training_space/* /tmp/pharmarl-h200/
   cp -r scripts server pyproject.toml /tmp/pharmarl-h200/
   cd /tmp/pharmarl-h200
   git add -A && git commit -m "deploy h200" && git push
   ```

2. **Set secrets in the Space settings**:
   - `PHARMARL_ENV_URL` = `https://anshumanatrey-pharmarl.hf.space` (the deployed env Space)
   - `HF_TOKEN` = a write-scoped HF token
   - `WANDB_API_KEY` = (optional) your W&B key
   - `HF_REPO` = `YOUR-USER/pharmarl-llama-3b-trained-vijay-h200`
   - `MAX_STEPS` = `200` (default; the issue's reference is 200)

3. **Confirm hardware = H200** in Space settings → Hardware. Frontmatter
   above already requests it; double-check the Space billing page reads
   `H200 ($5/hr)`. Set timeout = 4 h.

4. The Space starts, training runs, LoRA pushes to Hub, Space sits idle until
   you delete it. Total cost for a 200-step run: ~$8.50.

5. **Tear down when done**: Settings → Delete Space (or downgrade hardware to
   "CPU basic" to stop billing while keeping the trained adapter URL).

## Cost ceiling

H200 at $5/hr × 1.7h = **$8.50** expected. Hard cap with 4 h timeout = **$20**.

## Fallback if the H200 image fails to build

Drop back to a100-large with the older proven image
(`nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` + torch 2.4). Wall time goes
to ~2.8 h, cost ≈ $4 × 2.8 = $11.20, but the stack is battle-tested. The
older `Dockerfile` for that path lives on `main`.
