---
title: PharmaRL Training (Fallback)
emoji: 🧪
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
hardware: t4-small
pinned: false
license: bsd-3-clause
---

# PharmaRL training Space — Path C fallback

**This Space exists as a backup for if Colab dies during training.** It is NOT
the primary training path. Sahil's free Colab is. Spin this up only when:

- Colab disconnects three times in a row
- Colab GPU quota is exhausted
- HF deadline is <8h away and Colab is unstable

## What it does

On startup, runs `scripts/train_grpo.py` against the deployed env URL passed
via the `PHARMARL_ENV_URL` Space secret. SFT format priming → GRPO training →
pushes the trained LoRA to HF Hub. A minimal HTTP server stays up so the
Space doesn't show as "Crashed."

## How to deploy (5 minutes when you need it)

1. **Create the Space**:
   ```bash
   # From the pharmarl repo root
   huggingface-cli repo create pharmarl-training --type space --space_sdk docker
   git clone https://huggingface.co/spaces/YOUR-USER/pharmarl-training /tmp/pharmarl-training
   cp -r training_space/* /tmp/pharmarl-training/
   cp -r scripts server pyproject.toml /tmp/pharmarl-training/
   cd /tmp/pharmarl-training
   git add -A && git commit -m "deploy" && git push
   ```

2. **Set secrets in the Space settings**:
   - `PHARMARL_ENV_URL` = your env Space URL (e.g. `https://YOUR-USER-pharmarl.hf.space`)
   - `HF_TOKEN` = a write-scoped HF token
   - `WANDB_API_KEY` = (optional) your W&B key
   - `HF_REPO` = `YOUR-USER/pharmarl-llama-trained`
   - `MAX_STEPS` = `200` (default; bump if more time)

3. **Upgrade hardware to T4 Small** in Space settings → Hardware. Confirms
   ~$0.40/hr billing against your $30 hackathon credit.

4. The Space starts, training runs, LoRA pushes to Hub, Space sits idle until
   you delete it. Total cost for a 200-step run: ~$2-3 of credit.

5. **Tear down when done**: Settings → Delete Space (or downgrade hardware to
   "CPU basic" to stop billing while keeping the trained adapter URL).

## Cost ceiling

T4 Small at $0.40/hr × 6h = **$2.40**. If you run for 12h: **$4.80**.
Your $30 budget covers ~75 hours of T4 — far more than needed.

## What if I want faster?

Upgrade hardware to A10G Small ($1.05/hr) — roughly 2× throughput on the
forward passes. 200 GRPO steps drops from ~6h to ~3h. Cost: ~$3.

Don't use A100 ($4.13/hr) — overkill for a 1B model.
