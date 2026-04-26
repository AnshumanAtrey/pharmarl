# PharmaRL on HF Jobs (H200) — independent run

Self-contained training pipeline for issue #14: **H200 GPU on Hugging Face
Jobs**, no shared infrastructure with Anshuman's a10g-large run, no external
env Space, full audit trail saved to a Hugging Face model repo.

## Why HF Jobs (not Spaces)

Spaces are great for hosting persistent services (the env server is one).
For a one-shot training run we want a different shape:

| Concern | HF Spaces (h200 hardware) | HF Jobs (h200 flavor) |
|---|---|---|
| Billed during image build | Yes — H200 minutes burn while installing deps | No — only Starting/Running counts |
| Lifecycle | Long-lived; you have to remember to downgrade | Container exits, billing stops |
| Concurrency | One Space, one image at a time | Many parallel jobs, isolated logs |
| Native log API | Only Space logs UI | `hf jobs logs/inspect/ps`, Python SDK |
| Cancel mid-run | Manual hardware downgrade | `hf jobs cancel <id>` |

The H200 flavor on HF Jobs is **141 GB VRAM, $5/hr, 23 vCPU, 256 GB RAM** —
identical hardware to the H200 Space tier, just billed by the second.

## End-to-end flow

```
your laptop                  HF Jobs (h200 container)             HF Hub
─────────                    ──────────────────────────           ──────
launch_h200_job.py launch ─▶ pip install Hopper stack
                             git clone vijay-h200 branch ◀────── vijay2776/pharmarl
                             uvicorn server.app on :8000  
                             python scripts/h200_train_entry.py
                                ├─ GPU sanity → entry.log
                                ├─ run train_grpo.py ─────────▶  W&B (live curves)
                                ├─ checkpoints every 25 steps
                                └─ on done:                       vijay2776/pharmarl-llama-3b-trained-vijay-h200
                                   - upload lora_final/             ├─ adapter weights
                                   - upload logs/*                  ├─ logs/{entry,train,env_server}.log
                                   - upload run_metadata.json       ├─ logs/run_metadata.json
                                   - upload RUN_SUMMARY.md          └─ runs/<JOB_ID>/RUN_SUMMARY.md
```

## Prerequisites (one-time)

1. **Pro / Team / Enterprise account** — HF Jobs requires it. Confirm at https://huggingface.co/settings/jobs.

2. **CLI installed**:
   ```bash
   pip install -U huggingface_hub  # >=1.8.0 for run_job + flavors
   curl -LsSf https://hf.co/cli/install.sh | bash    # or: brew install hf
   hf auth login                                       # write-scope token
   ```

3. **Push the code branch** (already done):
   ```bash
   # On the inner pharmarl/ repo:
   git checkout vijay-h200
   # And the HF Space mirror at vijay2776/pharmarl already has this branch.
   ```

4. **(Optional) W&B**:
   ```bash
   export WANDB_API_KEY=...      # your W&B key — only needed if you pass --with-wandb
   ```

## Launch

```bash
# default 200-step GRPO run, ~1.7h on H200, ~$8.50
python scripts/launch_h200_job.py launch --with-wandb

# dry-run first if you want to see the bootstrap command:
python scripts/launch_h200_job.py launch --dry-run

# different model / step budget:
python scripts/launch_h200_job.py launch \
    --model unsloth/Llama-3.2-3B-Instruct \
    --max-steps 300 \
    --timeout 3h
```

The launcher prints the job id and URL. Jobs run in the background — Ctrl+C
on log streaming does not kill the run.

## Manage

```bash
hf jobs ps -a --filter label=purpose=h200-grpo   # list our H200 runs
hf jobs logs <JOB_ID>                             # stream logs (re-runnable)
hf jobs inspect <JOB_ID>                          # full status JSON
hf jobs stats <JOB_ID>                            # CPU/mem/GPU usage
hf jobs cancel <JOB_ID>                           # stop billing
```

Same operations are exposed through `python scripts/launch_h200_job.py
{status,logs,cancel,…}` for scripting.

## What gets logged (judging-criteria-aligned)

The hackathon's "Showing Improvement in Rewards" criterion (20%) wants
*observable evidence of training progress*. The single H200 run produces:

| Surface | What's captured | Where to look |
|---|---|---|
| **W&B dashboard** | per-step `mean_reward`, `parse_rate`, per-component rewards (`qed`, `docking`, `sa`, `toxicity_clean`), action-type histogram, sample SMILES table every 25 steps, KL, policy loss, grad norm | wandb.ai run URL (printed in `hf jobs logs <id>` and saved to `run_metadata.json`) |
| **HF Job logs** | full trainer stdout — replayable via `hf jobs logs <id>` | HF Jobs UI |
| **Hub model repo** (permanent) | `lora_final/`, `lora_step_{25,50,...}/`, `logs/{entry,train,env_server}.log`, `logs/run_metadata.json`, `runs/<JOB_ID>/RUN_SUMMARY.md` | https://huggingface.co/vijay2776/pharmarl-llama-3b-trained-vijay-h200 |

The `RUN_SUMMARY.md` written into the Hub repo is the single page judges
can open to find: GPU info, package versions, W&B link, training time, exit
code, and links to every log file. No need to dig through job UIs.

## Acceptance signals (from issue #14)

Watch for these strings in `hf jobs logs <JOB_ID>`:

1. `GPU: NVIDIA H200  capability=(9, 0)`
2. `unsloth import OK (...)`
3. `transformers import OK (4.46.3)` `trl import OK (0.13.0)`
4. `[main] env=http://127.0.0.1:8000`
5. After ~5 min: `[grpo] step= 0 trivial mean=... parse=100%`
6. Every 25 steps: `[grpo] step=...` lines without errors
7. End: `[push] done — https://huggingface.co/...-vijay-h200`

## Fallback if H200 is unavailable in your account

```bash
# A100 80GB — 2.8h wall, ~$7
python scripts/launch_h200_job.py launch --flavor a100-large --timeout 4h
```

The bootstrap stack (transformers 4.46.3 + trl 0.13.0 + unsloth 2025.2.15)
runs on Ampere too; only the **bitsandbytes / xformers** pins were added
for sm_90 and they remain backwards-compatible with sm_80.
