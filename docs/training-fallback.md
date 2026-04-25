# Path C — Training failover runbook

**Primary path is Sahil's Colab notebook.** This file is the playbook for when
Colab fails. Follow it in order — the lower-numbered options are faster and
cheaper. Do not skip ahead unless the option above is exhausted.

## Trigger conditions — when to bail on Colab

You're not on Plan B until at least one of:
- ❌ Colab disconnected three times in a row mid-training
- ❌ Colab GPU quota error: *"You cannot currently connect to a GPU due to usage limits"*
- ❌ The notebook OOMs even with `NUM_GENERATIONS = 4` and `MAX_NEW_TOKENS = 40`
- ⏰ ≤ 8h to submission and you don't have a clean reward curve yet

If none of those, **stay on Colab**. Don't burn HF credits chasing imaginary problems.

---

## Option 1 — Another teammate's free Colab (try first, ~5 min)

**Cost: $0. Cadence: instant.**

1. Anshuman or Vijay logs into a *different* Google account
2. Opens the same `colab/train_pharmarl.ipynb`
3. Sets `os.environ['PHARMARL_ENV_URL']` to the deployed env Space URL
4. Runs all cells

This works because Colab quotas are *per-account*, not per-team. Three teammates = three independent T4 quotas.

**Stop here if it works.**

---

## Option 2 — HF Spaces GPU training Space (~15 min to deploy + ~3-6h training)

**Cost: $2-5 of $30 hackathon credit. Cadence: 15 min to live.**

The repo includes a pre-built training Space template at `training_space/`.

### Deploy

```bash
# From the pharmarl repo root, with HF CLI logged in (huggingface-cli login)
HF_USER=$(huggingface-cli whoami | head -1)
SPACE=pharmarl-training

# Create the Space (Docker SDK)
huggingface-cli repo create "$SPACE" --type space --space_sdk docker -y

# Clone, copy training files, push
git clone "https://huggingface.co/spaces/$HF_USER/$SPACE" /tmp/$SPACE
cp -r training_space/* /tmp/$SPACE/
cp -r scripts server pyproject.toml /tmp/$SPACE/
cd /tmp/$SPACE
git add -A
git commit -m "Deploy PharmaRL training Space"
git push
```

### Configure secrets (HF Space → Settings → Variables and secrets)

| Key | Value |
|---|---|
| `PHARMARL_ENV_URL` | `https://YOUR-USER-pharmarl.hf.space` (your env Space) |
| `HF_TOKEN` | Write-scoped token from https://huggingface.co/settings/tokens |
| `HF_REPO` | `YOUR-USER/pharmarl-llama-trained` |
| `MAX_STEPS` | `200` (for 6h budget) or `100` (for 3h budget) |
| `WANDB_API_KEY` | (optional) |

### Upgrade hardware

HF Space → Settings → Hardware → **T4 Small ($0.40/hr)**.

### Watch progress

Hit `https://YOUR-USER-pharmarl-training.hf.space/status` — returns the last 30 lines of training log + current phase.

When `phase == "completed"`, the trained adapter is on HF Hub at the `HF_REPO` URL.

### Tear down

After training completes and you've confirmed the LoRA is on Hub:
- HF Space → Settings → Hardware → **CPU basic** (stops billing)
- Or: HF Space → Settings → Delete this Space

**Cost ceiling check**: T4 Small × 6h = $2.40. Even if you forget to tear down for 24h, that's $9.60 — well under the $30 budget.

---

## Option 3 — Run the script anywhere with a GPU (~30 min, ~$5)

**Cost: $0–10 depending on platform. Cadence: 30 min.**

Use this if HF Spaces deployment hits a Docker build issue and you're out of time. The same script (`scripts/train_grpo.py`) runs anywhere.

### Modal (recommended — has free credits)

```python
# modal_train.py
import modal

app = modal.App("pharmarl-train")
image = (modal.Image.debian_slim(python_version="3.10")
         .pip_install("torch==2.4.0", index_url="https://download.pytorch.org/whl/cu121")
         .pip_install("unsloth", "trl>=0.11.0", "transformers>=4.40.0", "peft",
                      "accelerate", "openenv-core", "selfies", "rdkit-pypi",
                      "PyTDC", "wandb", "huggingface_hub", "requests"))

@app.function(image=image, gpu="T4", timeout=21600,
              secrets=[modal.Secret.from_name("hf-token"),
                       modal.Secret.from_name("wandb-key")])
def train():
    import subprocess, os
    subprocess.run([
        "python", "-m", "scripts.train_grpo",
        "--env-url", os.environ["PHARMARL_ENV_URL"],
        "--max-steps", "200",
        "--hf-repo", "YOUR-USER/pharmarl-llama-trained",
        "--hf-token", os.environ["HF_TOKEN"],
    ], check=True)

if __name__ == "__main__":
    app.run()
```

```bash
pip install modal
modal token new
modal secret create hf-token HF_TOKEN=hf_...
modal secret create wandb-key WANDB_API_KEY=...
modal run modal_train.py
```

### Lambda Labs / RunPod / Vast.ai

```bash
# On the rented GPU box:
git clone https://github.com/AnshumanAtrey/pharmarl
cd pharmarl
pip install -e .
export HF_TOKEN=hf_...
export PHARMARL_ENV_URL=https://YOUR-USER-pharmarl.hf.space
python -m scripts.train_grpo \
    --env-url "$PHARMARL_ENV_URL" \
    --max-steps 200 \
    --hf-repo YOUR-USER/pharmarl-llama-trained \
    --hf-token "$HF_TOKEN"
```

T4 instances on these platforms run **$0.40–0.80/hr**. RunPod sometimes has community T4s for $0.20/hr.

---

## What about the rest of the team while training is running?

- **Vijay**: Records 90s pitch video using the script template at `docs/pitch-script.md`. Leave the *numbers* blank — fill them after training completes.
- **Anshuman**: Run `examples/demo.py --policies random scripted` against the deployed env. This generates the **baseline numbers** that go in the demo table — don't need a trained model for this.
- **You**: Babysit the training run. Hit `/status` every 30 min. Don't add features.

## Decision tree (read this when something breaks)

```
Colab disconnected mid-run?
├── Yes → Was a checkpoint saved at SAVE_EVERY?
│   ├── Yes → Resume from /content/pharmarl_step_N (easier, free)
│   └── No  → Lower MAX_STEPS, restart on different teammate's Colab (Option 1)
│
Colab GPU quota exhausted?
├── Yes → Option 1 (different teammate account)
│         If all 3 teammates exhausted → Option 2 (HF GPU Space)
│
Colab OOMs even at minimum settings?
├── Yes → Option 2 with T4 Small (more headroom than free Colab T4)
│         Or: Option 3 with A10G ($1.05/hr, 24GB VRAM)
│
≤ 4h to deadline + no curve yet?
├── Yes → Option 3 with A10G (faster wall-clock)
│         Cap MAX_STEPS at 100 — a clear curve on 100 steps beats no curve on 200
```

## Honest cost ceilings

| Path | Cost | Deploy time | Training time (200 steps) |
|---|---|---|---|
| Sahil's Colab | $0 | 5 min | 4–6 h |
| Other teammate's Colab | $0 | 5 min | 4–6 h |
| HF Space T4 Small | $2–5 | 15 min | 4–6 h |
| HF Space A10G Small | $3–8 | 15 min | 2–3 h |
| Modal T4 | $2–4 | 30 min | 4–6 h |
| Modal A10G | $4–7 | 30 min | 2–3 h |
| RunPod T4 community | $1–2 | 30 min | 4–6 h |

**Total $30 HF credit budget covers ~75 hours of T4 Small.** You will not run out unless you forget to tear down a Space.

## Final reminder

The most expensive thing isn't compute — it's *time spent switching paths*. Every 30 min you spend setting up Plan B is 30 min not spent on the demo video, README polish, or pitch rehearsal. Bail on Colab only when you have to, not when you're nervous.
