"""H200 training entrypoint — runs entirely inside an HF Job container.

Self-contained run, no Anshuman / external Space dependency:

  1. Captures host metadata + GPU sanity (issue #14 acceptance signals)
  2. Verifies the Hopper-pinned stack imports (unsloth / transformers / trl /
     bitsandbytes / xformers) before spending H200 minutes on training
  3. Pre-warms TDC oracles so the first /step doesn't stall ~30s
  4. Starts the FastAPI env server in-process on 127.0.0.1:8000
  5. Runs `python -m scripts.train_grpo` against the local env
  6. Uploads the full audit trail (raw stdout, env log, run metadata,
     W&B run id) to the HF Hub model repo alongside the trained adapter

The point of doing this in a single container instead of pointing at the
shared env Space:
  - The judges' "evidence of training progress" criterion (20%) needs
    metrics that survive after the Job ends. W&B + a frozen JSON in the
    Hub repo is a permanent record. Sharing infra with another team's
    Space leaves the run vulnerable to their lifecycle.
  - The env oracles are CPU-bound (TDC classifiers + RDKit). Running the
    server inline on the H200's 23 vCPUs is faster than HTTP across the
    public internet — no rate-limit, no cold-starts, no shared CPU contention.

Env vars expected (set via `hf jobs run --secrets / -e`):
  HF_TOKEN          (secret, required) — write-scope token for HF Hub push
  HF_REPO           (env, required)    — model repo to receive adapter + logs
  WANDB_API_KEY     (secret, optional) — enables live reward curves
  WANDB_PROJECT     (env, optional)    — defaults to "pharmarl"
  WANDB_RUN_NAME    (env, optional)    — defaults to "h200-{JOB_ID}"
  MAX_STEPS         (env, optional)    — defaults to 200
  NUM_GENERATIONS   (env, optional)    — defaults to 8
  SFT_WARMUP_STEPS  (env, optional)    — defaults to 60
  MODEL             (env, optional)    — defaults to unsloth/Llama-3.2-1B-Instruct
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

LOG_DIR = Path("/tmp/h200_run")
LOG_DIR.mkdir(parents=True, exist_ok=True)
ENTRY_LOG = LOG_DIR / "entry.log"
ENV_LOG = LOG_DIR / "env_server.log"
TRAIN_LOG = LOG_DIR / "train.log"
META_PATH = LOG_DIR / "run_metadata.json"


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    line = f"[{_stamp()}] [h200-entry] {msg}"
    print(line, flush=True)
    with ENTRY_LOG.open("a") as f:
        f.write(line + "\n")


def fail(msg: str, code: int = 2) -> None:
    log(f"FATAL: {msg}")
    sys.exit(code)


# ─── 1. Host metadata ────────────────────────────────────────────────────

JOB_ID = os.environ.get("JOB_ID", "local")
ACCELERATOR = os.environ.get("ACCELERATOR", "unknown")
META: dict = {
    "job_id": JOB_ID,
    "accelerator": ACCELERATOR,
    "cpu_cores": os.environ.get("CPU_CORES"),
    "memory": os.environ.get("MEMORY"),
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model": os.environ.get("MODEL", "unsloth/Llama-3.2-1B-Instruct"),
    "max_steps": int(os.environ.get("MAX_STEPS", "200")),
    "num_generations": int(os.environ.get("NUM_GENERATIONS", "8")),
    "sft_warmup_steps": int(os.environ.get("SFT_WARMUP_STEPS", "60")),
}

log(f"JOB_ID={JOB_ID} ACCELERATOR={ACCELERATOR}")
log(f"config: model={META['model']} max_steps={META['max_steps']} "
    f"G={META['num_generations']} sft={META['sft_warmup_steps']}")

HF_REPO = os.environ.get("HF_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_REPO:
    fail("HF_REPO env var is required (e.g. vijay2776/pharmarl-llama-3b-trained-vijay-h200)")
if not HF_TOKEN:
    fail("HF_TOKEN secret is required for HF Hub push")
META["hf_repo"] = HF_REPO

# ─── 2. GPU + stack sanity ───────────────────────────────────────────────

try:
    import torch  # noqa: E402
except Exception as e:
    fail(f"torch import failed: {e}")

if not torch.cuda.is_available():
    fail("CUDA not available — H200 flavor expected, got CPU-only runtime")

gpu_name = torch.cuda.get_device_name(0)
gpu_cap = tuple(torch.cuda.get_device_capability(0))
META["gpu"] = {"name": gpu_name, "capability": list(gpu_cap),
               "count": torch.cuda.device_count(),
               "torch_version": torch.__version__,
               "torch_cuda": torch.version.cuda}
log(f"GPU: {gpu_name}  capability={gpu_cap}  count={torch.cuda.device_count()}")
log(f"torch={torch.__version__} cuda={torch.version.cuda}")
if gpu_cap[0] != 9:
    log(f"WARNING: expected sm_90 (Hopper); got sm_{gpu_cap[0]}{gpu_cap[1]}")

for mod_name in ("transformers", "trl", "unsloth", "bitsandbytes", "xformers", "peft", "accelerate"):
    try:
        m = __import__(mod_name)
        ver = getattr(m, "__version__", "?")
        log(f"{mod_name} import OK ({ver})")
        META.setdefault("versions", {})[mod_name] = ver
    except Exception as e:
        fail(f"{mod_name} import failed: {e}")

# Oracle pre-warm now happens via HTTP after the env server is up — see below.

# ─── 4. Start env server sidecar ─────────────────────────────────────────

log("starting env server: uvicorn server.app:app on 127.0.0.1:8000")
env_proc = subprocess.Popen(
    ["uvicorn", "server.app:app", "--host", "127.0.0.1",
     "--port", "8000", "--log-level", "warning"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)
META["env_pid"] = env_proc.pid


def _drain_env_log():
    with ENV_LOG.open("a") as f:
        for line in env_proc.stdout:  # type: ignore[union-attr]
            f.write(line)
            f.flush()


threading.Thread(target=_drain_env_log, daemon=True).start()


def _wait_env_ready(url: str = "http://127.0.0.1:8000", timeout: int = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if env_proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


if not _wait_env_ready():
    log("env server failed to come up — last 80 lines of env_server.log:")
    if ENV_LOG.exists():
        for line in ENV_LOG.read_text().splitlines()[-80:]:
            print(line, flush=True)
    env_proc.kill()
    fail("env health check timed out", code=3)
log("env health OK at http://127.0.0.1:8000/health")

# Pre-warm oracles via HTTP — first /step on each target downloads the TDC
# classifier (~30MB each). Doing it now keeps the first GRPO rollout fast.
log("pre-warming oracles via /reset+/step…")
try:
    import requests
    requests.post("http://127.0.0.1:8000/reset", json={"difficulty": "trivial"}, timeout=60)
    for _ in range(2):
        requests.post(
            "http://127.0.0.1:8000/step",
            json={"action_type": "ADD_FRAGMENT", "fragment": "C", "position": 0},
            timeout=120,
        )
    log("oracle pre-warm done")
except Exception as e:
    log(f"WARNING: oracle pre-warm hit error (continuing): {e}")

# Make sure the env subprocess dies with us.
def _shutdown(signum, frame):  # noqa: ARG001
    log(f"signal {signum} received — terminating env subprocess")
    try:
        env_proc.terminate()
        env_proc.wait(timeout=5)
    except Exception:
        env_proc.kill()
    sys.exit(130)


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)

# ─── 5. Run trainer ──────────────────────────────────────────────────────

# Default WandB run name to JOB_ID so the W&B URL is grep-able from the audit trail.
os.environ.setdefault("WANDB_PROJECT", "pharmarl")
os.environ.setdefault("WANDB_RUN_NAME", f"h200-{JOB_ID}")
META["wandb_project"] = os.environ["WANDB_PROJECT"]
META["wandb_run_name"] = os.environ["WANDB_RUN_NAME"]

cmd = [
    sys.executable, "-m", "scripts.train_grpo",
    "--env-url", "http://127.0.0.1:8000",
    "--model", META["model"],
    "--max-steps", str(META["max_steps"]),
    "--num-generations", str(META["num_generations"]),
    "--sft-warmup-steps", str(META["sft_warmup_steps"]),
    "--output-dir", "/data/trained",
    "--hf-repo", HF_REPO,
    "--hf-token", HF_TOKEN,
    "--save-every", os.environ.get("SAVE_EVERY", "25"),
    "--audit-every", os.environ.get("AUDIT_EVERY", "25"),
    # The base image ships torch+cu124; pip-installing vllm on top tends to
    # pull cu13 nvidia wheels and break the runtime. Skip vllm entirely —
    # H200 is fast enough that eager inference still meets the 4h budget.
    "--no-fast-inference",
]
log(f"launching trainer: {' '.join(c for c in cmd if not c.startswith('hf_'))}")

t0 = time.time()
with TRAIN_LOG.open("a") as logf:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    # Tee trainer output to both Job logs (visible via `hf jobs logs <id>`)
    # and a file we can upload to the Hub afterwards.
    for line in proc.stdout:  # type: ignore[union-attr]
        sys.stdout.write(line)
        sys.stdout.flush()
        logf.write(line)
        logf.flush()
    proc.wait()
    rc = proc.returncode
elapsed = time.time() - t0
META["training_seconds"] = round(elapsed, 1)
META["trainer_exit_code"] = rc
META["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
log(f"trainer exit={rc} elapsed={elapsed:.0f}s ({elapsed/60:.1f}min)")

# Try to capture the W&B run URL from the trainer's stdout so judges can
# click straight to the live curves from the Hub repo.
wandb_url = None
try:
    import re
    text = TRAIN_LOG.read_text()
    m = re.search(r"https://wandb\.ai/[^\s]+", text)
    if m:
        wandb_url = m.group(0).rstrip(".,)")
        META["wandb_run_url"] = wandb_url
        log(f"detected W&B run: {wandb_url}")
except Exception:
    pass

# ─── 6. Upload audit trail to HF Hub ─────────────────────────────────────

META_PATH.write_text(json.dumps(META, indent=2, default=str))
log(f"run metadata written → {META_PATH}")

try:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO, exist_ok=True, repo_type="model")
    uploads: list[tuple[Path, str]] = [
        (META_PATH, "logs/run_metadata.json"),
        (ENTRY_LOG, "logs/entry.log"),
        (TRAIN_LOG, "logs/train.log"),
        (ENV_LOG, "logs/env_server.log"),
    ]
    for src, dst in uploads:
        if not src.exists() or src.stat().st_size == 0:
            log(f"skip upload (missing/empty): {src}")
            continue
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dst,
            repo_id=HF_REPO,
            repo_type="model",
            commit_message=f"H200 run {JOB_ID}: {dst}",
        )
        log(f"uploaded {src.name} → {HF_REPO}:{dst}")

    # Also drop a human-readable RUN_SUMMARY.md at the repo root so anyone
    # opening the Hub page sees the audit trail immediately.
    summary_lines = [
        f"# H200 GRPO Run — {JOB_ID}",
        "",
        f"- **Job:** https://huggingface.co/jobs/{os.environ.get('USER', '?')}/{JOB_ID}",
        f"- **Started:** {META['started_at']}",
        f"- **Finished:** {META.get('finished_at', '—')}",
        f"- **Wall time:** {META.get('training_seconds', '—')}s",
        f"- **Trainer exit code:** {rc}",
        f"- **Accelerator:** {ACCELERATOR}",
        f"- **GPU:** {gpu_name} (capability {gpu_cap})",
        f"- **Model:** {META['model']}",
        f"- **GRPO steps:** {META['max_steps']} | G={META['num_generations']} | SFT warmup={META['sft_warmup_steps']}",
        "",
        "## Stack",
        "",
        *[f"- `{k}=={v}`" for k, v in META.get("versions", {}).items()],
        "",
        "## Live metrics",
        "",
        (f"- W&B run: {wandb_url}" if wandb_url else "- W&B: (not configured for this run)"),
        "",
        "## Audit trail (in this repo)",
        "",
        "- [`logs/entry.log`](logs/entry.log) — host setup + sanity checks",
        "- [`logs/train.log`](logs/train.log) — full trainer stdout (per-step rewards, parse rate, W&B URL)",
        "- [`logs/env_server.log`](logs/env_server.log) — env sidecar log",
        "- [`logs/run_metadata.json`](logs/run_metadata.json) — machine-readable run metadata",
        "- Adapter: `lora_final/` (pushed by `train_grpo.py` itself)",
    ]
    summary_path = LOG_DIR / "RUN_SUMMARY.md"
    summary_path.write_text("\n".join(summary_lines))
    api.upload_file(
        path_or_fileobj=str(summary_path),
        path_in_repo=f"runs/{JOB_ID}/RUN_SUMMARY.md",
        repo_id=HF_REPO,
        repo_type="model",
        commit_message=f"H200 run {JOB_ID}: summary",
    )
    log(f"uploaded RUN_SUMMARY.md → {HF_REPO}:runs/{JOB_ID}/RUN_SUMMARY.md")
except Exception as e:
    log(f"WARNING: hub upload failed: {e}  (logs are still in `hf jobs logs {JOB_ID}`)")

# ─── 7. Cleanup ──────────────────────────────────────────────────────────

try:
    env_proc.terminate()
    env_proc.wait(timeout=5)
except Exception:
    env_proc.kill()

log(f"done — exit {rc}")
sys.exit(rc)
