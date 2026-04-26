"""Training Space entrypoint: heartbeat HTTP server + background training job.

HF Spaces require a process listening on `app_port` (7860). We satisfy that
with a tiny FastAPI server exposing `/status` (last training log line) and
`/metrics` (basic counts). Training runs in a daemon thread; logs are
appended to /tmp/train.log and the most recent lines are served.

Configuration is via Space secrets:
  PHARMARL_ENV_URL  (required) — URL of the deployed env Space
  HF_TOKEN          (required) — write-scoped token for adapter push
  HF_REPO           (required) — destination repo id
  MAX_STEPS         (default 200)
  MODEL             (default unsloth/Llama-3.2-1B-Instruct)
  NUM_GENERATIONS   (default 8)
  WANDB_API_KEY     (optional)
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI
import uvicorn

LOG_PATH = Path("/tmp/train.log")
STATE = {"phase": "starting", "started_at": time.time(), "exit_code": None}


def _required(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        STATE["phase"] = f"error: missing {key}"
        raise RuntimeError(f"Required env var {key!r} not set")
    return val


def _gpu_sanity_check(logf) -> None:
    """Issue #14 acceptance: log GPU + unsloth import OK before training starts."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            logf.write(f"[{time.strftime('%H:%M:%S')}] GPU: {name} capability={cap}\n")
            STATE["gpu"] = name
            STATE["gpu_capability"] = list(cap)
        else:
            logf.write(f"[{time.strftime('%H:%M:%S')}] GPU: NONE (cuda.is_available()=False)\n")
            STATE["gpu"] = None
        import unsloth  # noqa: F401
        logf.write(f"[{time.strftime('%H:%M:%S')}] unsloth import OK\n")
        STATE["unsloth_ok"] = True
    except Exception as e:
        logf.write(f"[{time.strftime('%H:%M:%S')}] sanity check FAILED: {e}\n")
        STATE["unsloth_ok"] = False
    logf.flush()


def _train_in_background():
    try:
        env_url = _required("PHARMARL_ENV_URL")
        hf_token = _required("HF_TOKEN")
        hf_repo = _required("HF_REPO")
        model = os.environ.get("MODEL", "unsloth/Llama-3.2-1B-Instruct")
        max_steps = int(os.environ.get("MAX_STEPS", "200"))
        num_gen = int(os.environ.get("NUM_GENERATIONS", "8"))
        sft_steps = int(os.environ.get("SFT_WARMUP_STEPS", "60"))

        cmd = [
            "python", "-m", "scripts.train_grpo",
            "--env-url", env_url,
            "--model", model,
            "--max-steps", str(max_steps),
            "--num-generations", str(num_gen),
            "--sft-warmup-steps", str(sft_steps),
            "--output-dir", "/data/trained",
            "--hf-repo", hf_repo,
            "--hf-token", hf_token,
        ]
        STATE["phase"] = "training"
        STATE["cmd"] = " ".join(cmd)
        with open(LOG_PATH, "a") as logf:
            _gpu_sanity_check(logf)
            logf.write(f"[{time.strftime('%H:%M:%S')}] starting: {' '.join(cmd)}\n")
            logf.flush()
            proc = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT,
                cwd="/app", env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            STATE["pid"] = proc.pid
            proc.wait()
            STATE["exit_code"] = proc.returncode
            STATE["phase"] = "completed" if proc.returncode == 0 else "failed"
            logf.write(f"[{time.strftime('%H:%M:%S')}] exit={proc.returncode}\n")
    except Exception as e:
        STATE["phase"] = f"crashed: {e}"
        with open(LOG_PATH, "a") as logf:
            logf.write(f"[{time.strftime('%H:%M:%S')}] CRASH: {e}\n")


app = FastAPI(title="PharmaRL Training Space")


@app.get("/")
def root():
    return {
        "service": "pharmarl-training",
        "phase": STATE["phase"],
        "elapsed_seconds": int(time.time() - STATE["started_at"]),
        "exit_code": STATE["exit_code"],
    }


@app.get("/status")
def status():
    tail = ""
    if LOG_PATH.exists():
        try:
            tail = LOG_PATH.read_text().splitlines()[-30:]
        except Exception as e:
            tail = [f"<error reading log: {e}>"]
    return {**STATE, "log_tail": tail}


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    threading.Thread(target=_train_in_background, daemon=True).start()
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
