"""One-shot fetcher for the anshuman-a10g run audit bundle.

Adapted from runs/vijay-h200-llama1b/_fetch.py — same shape, different IDs.

Pulls every artefact a teammate would need to verify the run, *without
requiring any wandb / HF auth on their end*:

  - W&B run history (per-step metrics) → wandb/run_history.csv + .json
  - W&B run summary (final values)     → wandb/run_summary.json
  - W&B run config                      → wandb/run_config.json
  - W&B system metrics (GPU/CPU)        → wandb/system_metrics.csv
  - HF Hub model repo log files         → hf_hub/{entry,train,env_server}.log
  - HF Hub run metadata                 → hf_hub/run_metadata.json
  - HF Hub run summary                  → hf_hub/RUN_SUMMARY.md
  - Static charts                        → plots/*.png

Run once after the HF Job finishes pushing the LoRA, commit the outputs,
push to GitHub. After that, anyone with the GitHub repo can read the
run bundle offline — no logins, no API calls.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WANDB_DIR = HERE / "wandb"
HF_DIR = HERE / "hf_hub"
PLOTS_DIR = HERE / "plots"

# ─── Run identifiers — UPDATE THESE WHEN TRAINING FINISHES ─────────────
# W&B run path is shown in the HF Job's tail logs as:
#   wandb: 🚀 View run <name> at: https://wandb.ai/<entity>/<project>/runs/<id>
# Take the path-after-wandb.ai/ portion.
WANDB_RUN_PATH = ""  # e.g. "anshumanatrey-itm/pharmarl/runs/abc123"

# HF Hub model repo where the trained LoRA was pushed by train_grpo.py.
# Set in the HF Job command via --hf-repo.
HF_HUB_REPO = "anshumanatrey/pharmarl-llama-3b-trained-anshuman"

# HF Job ID. Used to label plots and locate per-job log files in the model repo.
JOB_ID = "69ed99a9d2c8bd8662bcf2ef"


# ─── Load WANDB_API_KEY from project .env ────────────────────────────────

def load_dotenv() -> None:
    env_path = HERE.parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


load_dotenv()
if not WANDB_RUN_PATH:
    print("ERROR: WANDB_RUN_PATH is empty. Edit this file and paste the W&B run path", file=sys.stderr)
    print("       (find it in the HF Job tail logs: 'View run ... at: https://wandb.ai/<path>')", file=sys.stderr)
    sys.exit(1)
if not os.environ.get("WANDB_API_KEY"):
    print("ERROR: WANDB_API_KEY missing — set it in pharmarl/.env", file=sys.stderr)
    sys.exit(1)


# ─── W&B fetch ───────────────────────────────────────────────────────────

print(f"[wandb] connecting to run {WANDB_RUN_PATH}")
import wandb  # noqa: E402

api = wandb.Api()
run = api.run(WANDB_RUN_PATH)

print(f"[wandb] fetching history…")
history = run.history(samples=10000, pandas=True)
hist_csv = WANDB_DIR / "run_history.csv"
history.to_csv(hist_csv, index=False)
hist_json = WANDB_DIR / "run_history.json"
hist_json.write_text(history.to_json(orient="records", indent=2))
print(f"  → {hist_csv}  ({len(history)} rows, {len(history.columns)} cols)")

summary = {k: _v for k, _v in run.summary.items() if not k.startswith("_")}
sum_path = WANDB_DIR / "run_summary.json"
sum_path.write_text(json.dumps(summary, indent=2, default=str))
print(f"  → {sum_path}  ({len(summary)} keys)")

config = dict(run.config)
cfg_path = WANDB_DIR / "run_config.json"
cfg_path.write_text(json.dumps(config, indent=2, default=str))
print(f"  → {cfg_path}")

print(f"[wandb] fetching system metrics…")
sys_history = run.history(stream="systemMetrics", samples=10000, pandas=True)
sys_csv = WANDB_DIR / "system_metrics.csv"
sys_history.to_csv(sys_csv, index=False)
print(f"  → {sys_csv}  ({len(sys_history)} rows)")

meta = {
    "id": run.id,
    "name": run.name,
    "project": run.project,
    "entity": run.entity,
    "url": run.url,
    "state": run.state,
    "created_at": str(run.created_at),
    "tags": list(run.tags) if hasattr(run, "tags") else [],
    "summary_keys": list(summary.keys()),
    "history_columns": list(history.columns),
    "history_rows": len(history),
}
(WANDB_DIR / "run_meta.json").write_text(json.dumps(meta, indent=2))
print(f"  → wandb/run_meta.json")


# ─── HF Hub log files (public repo, no auth needed for download) ─────────

def hf_resolve(repo: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


hf_files = [
    "logs/entry.log",
    "logs/train.log",
    "logs/env_server.log",
    "logs/run_metadata.json",
    f"runs/{JOB_ID}/RUN_SUMMARY.md",
]

print(f"[hf_hub] downloading log files from {HF_HUB_REPO}")
for rel in hf_files:
    url = hf_resolve(HF_HUB_REPO, rel)
    out = HF_DIR / rel.split("/")[-1]
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            out.write_bytes(r.read())
        size = out.stat().st_size
        print(f"  → {out}  ({size:,} bytes)")
    except urllib.error.HTTPError as e:
        print(f"  ✗ {rel}: HTTP {e.code} (file may not be in repo — that's OK if not produced)")
    except Exception as e:
        print(f"  ✗ {rel}: {e}")


# ─── Plots — static PNGs anyone can open ─────────────────────────────────

print(f"[plots] generating PNGs")
import matplotlib.pyplot as plt  # noqa: E402

plt.style.use("dark_background")

def _save(fig, name: str) -> None:
    out = PLOTS_DIR / f"{name}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  → {out}")


step = history.get("step")
if step is None:
    print("[plots] no 'step' column — skipping plots")
else:
    if "mean_reward" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(step, history["mean_reward"], color="#10b981", linewidth=2, label="mean reward")
        if "max_reward" in history.columns:
            ax.plot(step, history["max_reward"], color="#22d3ee", linewidth=1, alpha=0.7, label="max reward")
        ax.set_xlabel("GRPO step")
        ax.set_ylabel("reward")
        ax.set_title("anshuman-a10g — reward over GRPO steps")
        ax.grid(alpha=0.2)
        ax.legend()
        _save(fig, "01-reward-mean-and-max-per-grpo-step")

    if "policy_loss" in history.columns and "kl" in history.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(step, history["policy_loss"], color="#a855f7", linewidth=1.5)
        ax1.set_xlabel("GRPO step"); ax1.set_ylabel("policy loss")
        ax1.set_title("policy loss")
        ax1.grid(alpha=0.2)
        ax2.plot(step, history["kl"], color="#f59e0b", linewidth=1.5)
        ax2.set_xlabel("GRPO step"); ax2.set_ylabel("KL divergence")
        ax2.set_title("KL — frozen-reference penalty")
        ax2.grid(alpha=0.2)
        _save(fig, "02-policy-loss-and-kl-divergence")

    if "parse_rate" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(step, history["parse_rate"], color="#10b981", linewidth=2)
        ax.fill_between(step, 0, history["parse_rate"], color="#10b981", alpha=0.18)
        ax.set_xlabel("GRPO step"); ax.set_ylabel("parse rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("parse rate — should be ~100% (3B + chat-template SYSTEM, no SFT warmup needed)")
        ax.grid(alpha=0.2)
        _save(fig, "03-parse-rate")

    comps = [c for c in ("reward_component/qed", "reward_component/docking",
                         "reward_component/sa", "reward_component/toxicity_clean")
             if c in history.columns]
    if comps:
        fig, ax = plt.subplots(figsize=(10, 4))
        for c, col in zip(comps, ["#10b981", "#06b6d4", "#f59e0b", "#a855f7"]):
            ax.plot(step, history[c], label=c.split("/")[1], color=col, linewidth=1.6)
        ax.set_xlabel("GRPO step"); ax.set_ylabel("oracle component (0–1)")
        ax.set_title("reward component decomposition — composite oracle per step")
        ax.grid(alpha=0.2); ax.legend()
        _save(fig, "04-oracle-components-qed-docking-sa-toxicity")

    if "loss" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(step, history["loss"], color="#22d3ee", linewidth=1.5)
        ax.set_xlabel("step"); ax.set_ylabel("loss")
        ax.set_title("GRPO total loss")
        ax.grid(alpha=0.2)
        _save(fig, "05-grpo-total-loss")

print(f"\n✅ done. Bundle ready in {HERE}")
print(f"   Run path: {WANDB_RUN_PATH}")
print(f"   HF repo:  https://huggingface.co/{HF_HUB_REPO}")
print(f"   HF Job:   https://huggingface.co/jobs/anshumanatrey/{JOB_ID}")
