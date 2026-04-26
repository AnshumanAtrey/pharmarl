"""Launch a self-contained PharmaRL H200 GRPO run on Hugging Face Jobs.

What this does, in order:

  1. Submits a Docker job to HF Jobs with --flavor h200 ($5/hr, 141GB VRAM).
  2. The container clones this repo's `vijay-h200` branch from the
     vijay2776/pharmarl HF Space (no GitHub or Anshuman dependency).
  3. Installs the Hopper-pinned stack (issue #14 acceptance versions).
  4. Hands off to scripts/h200_train_entry.py which runs the env sidecar +
     trainer in the same container and uploads logs/adapter to HF Hub.

Usage (from your laptop, after `hf auth login`):

  # default: 200 GRPO steps, ~1.7h on H200, ~$8.50
  python scripts/launch_h200_job.py launch

  # check what's running
  python scripts/launch_h200_job.py status

  # tail logs of a running job (Ctrl+C is safe — job keeps running)
  python scripts/launch_h200_job.py logs <JOB_ID>

  # cancel
  python scripts/launch_h200_job.py cancel <JOB_ID>

This is intentionally a thin wrapper around `huggingface_hub.run_job` —
all the cluster-side logic lives in scripts/h200_train_entry.py so it can
be re-used (and unit-tested) without going through HF Jobs.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path


def _load_dotenv() -> None:
    """Hydrate os.environ from ./.env (or pharmarl/.env) if it exists.

    Tiny dotenv reader — keeps the launcher dependency-free. Only sets keys
    that aren't already in the environment, so an explicit export still wins.
    """
    here = Path(__file__).resolve()
    for candidate in (Path.cwd() / ".env", here.parent.parent / ".env"):
        if not candidate.exists():
            continue
        for raw in candidate.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        break


_load_dotenv()

# The stack pin matches issue #14 ("xformers 0.0.27 lacks sm_90; bnb 0.45.5+
# adds Hopper LLM.int8"). Keep these versions in lockstep with
# training_space/Dockerfile.
PINNED_DEPS = [
    "transformers==4.46.3",
    "trl==0.13.0",
    "unsloth==2025.2.15",
    "bitsandbytes>=0.45.5",
    "xformers>=0.0.30",
    "peft",
    "accelerate",
    "openenv-core",
    "selfies",
    "rdkit-pypi",
    "PyTDC",
    "wandb",
    "huggingface_hub",
    "fastapi",
    "uvicorn",
    "requests",
    "setuptools<81",  # PyTDC 0.4.x relies on pkg_resources
]

DEFAULT_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
DEFAULT_FLAVOR = "h200"
DEFAULT_TIMEOUT = "4h"
DEFAULT_CODE_REPO = "https://huggingface.co/spaces/vijay2776/pharmarl"
DEFAULT_BRANCH = "vijay-h200"
DEFAULT_HF_REPO = "vijay2776/pharmarl-llama-3b-trained-vijay-h200"


def _build_command(code_repo: str, branch: str) -> list[str]:
    """Return the bash -c command the H200 container will execute."""
    deps = " ".join(f'"{d}"' for d in PINNED_DEPS)
    script = textwrap.dedent(f"""\
        set -euo pipefail
        echo "[bootstrap] $(date -u +%FT%TZ)  JOB_ID=${{JOB_ID:-?}}  ACCELERATOR=${{ACCELERATOR:-?}}"
        export DEBIAN_FRONTEND=noninteractive
        echo "[bootstrap] apt-get install git build-essential libxrender1 libxext6 libsm6"
        apt-get update -qq
        apt-get install -y --no-install-recommends git build-essential libxrender1 libxext6 libsm6
        echo "[bootstrap] pip install pinned stack"
        python -m pip install --upgrade pip
        python -m pip install --no-cache-dir {deps}
        echo "[bootstrap] git clone -b {branch} {code_repo} /app"
        git clone --depth=1 -b {branch} {code_repo} /app
        cd /app
        echo "[bootstrap] handing off to h200_train_entry.py"
        exec python scripts/h200_train_entry.py
    """)
    return ["bash", "-c", script]


def cmd_launch(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import run_job
    except ImportError:
        print("ERROR: install huggingface_hub>=1.8.0  (pip install -U huggingface_hub)",
              file=sys.stderr)
        return 1

    env = {
        "HF_REPO": args.hf_repo,
        "MAX_STEPS": str(args.max_steps),
        "NUM_GENERATIONS": str(args.num_generations),
        "SFT_WARMUP_STEPS": str(args.sft_warmup_steps),
        "MODEL": args.model,
        "WANDB_PROJECT": args.wandb_project,
        # SAVE_EVERY/AUDIT_EVERY override defaults inside the entry script
        "SAVE_EVERY": str(args.save_every),
        "AUDIT_EVERY": str(args.audit_every),
    }
    secrets = {"HF_TOKEN": True}  # passed implicitly from local hf auth
    if args.with_wandb:
        wandb_key = os.environ.get("WANDB_API_KEY")
        if not wandb_key:
            print("WARNING: --with-wandb set but WANDB_API_KEY not in your shell. "
                  "Run `export WANDB_API_KEY=...` first or omit the flag.",
                  file=sys.stderr)
        else:
            secrets["WANDB_API_KEY"] = wandb_key

    labels = {
        "project": "pharmarl",
        "owner": "vijay",
        "purpose": "h200-grpo",
        "branch": args.branch,
    }

    print(f"[launch] image:   {args.image}")
    print(f"[launch] flavor:  {args.flavor}  (timeout {args.timeout})")
    print(f"[launch] code:    {args.code_repo}@{args.branch}")
    print(f"[launch] hf_repo: {args.hf_repo}")
    print(f"[launch] env:     MAX_STEPS={env['MAX_STEPS']} G={env['NUM_GENERATIONS']} "
          f"SFT={env['SFT_WARMUP_STEPS']} model={env['MODEL']}")
    print(f"[launch] secrets: {sorted(secrets)}")
    if args.dry_run:
        print("[launch] --dry-run: not submitting")
        print("[launch] command would be:")
        for line in _build_command(args.code_repo, args.branch)[-1].splitlines():
            print(f"  {line}")
        return 0

    job = run_job(
        image=args.image,
        command=_build_command(args.code_repo, args.branch),
        env=env,
        secrets=secrets,
        flavor=args.flavor,
        timeout=args.timeout,
        labels=labels,
    )
    print(f"\n[launch] submitted ✔")
    print(f"[launch] job_id:  {job.id}")
    print(f"[launch] url:     {job.url}")
    print()
    print("Tail logs:")
    print(f"  hf jobs logs {job.id}")
    print("List your H200 jobs:")
    print("  hf jobs ps -a --filter label=purpose=h200-grpo")
    print("Cancel:")
    print(f"  hf jobs cancel {job.id}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from huggingface_hub import list_jobs
    jobs = list_jobs()
    rows = [j for j in jobs if (j.environment or {}).get("HF_REPO", "").endswith("vijay-h200")
            or any(l in (getattr(j, "labels", {}) or {}).items()
                   for l in [("purpose", "h200-grpo")])]
    if not rows:
        # Fallback: show all your jobs.
        rows = list(jobs)[:20]
    print(f"{'JOB_ID':<28} {'FLAVOR':<14} {'STATUS':<12} CREATED")
    for j in rows:
        status = j.status.stage if j.status else "?"
        print(f"{j.id:<28} {j.flavor:<14} {status:<12} {j.created_at}")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    from huggingface_hub import fetch_job_logs
    for line in fetch_job_logs(job_id=args.job_id):
        sys.stdout.write(line if line.endswith("\n") else line + "\n")
        sys.stdout.flush()
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    from huggingface_hub import cancel_job
    cancel_job(job_id=args.job_id)
    print(f"cancelled {args.job_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # launch
    pl = sub.add_parser("launch", help="submit a new H200 training job")
    pl.add_argument("--image", default=DEFAULT_IMAGE)
    pl.add_argument("--flavor", default=DEFAULT_FLAVOR,
                    help=f"HF Jobs hardware flavor (default: {DEFAULT_FLAVOR})")
    pl.add_argument("--timeout", default=DEFAULT_TIMEOUT,
                    help=f"e.g. 4h, 90m (default: {DEFAULT_TIMEOUT})")
    pl.add_argument("--code-repo", default=DEFAULT_CODE_REPO)
    pl.add_argument("--branch", default=DEFAULT_BRANCH)
    pl.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                    help="HF model repo to receive adapter + audit logs")
    pl.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct")
    pl.add_argument("--max-steps", type=int, default=200)
    pl.add_argument("--num-generations", type=int, default=8)
    pl.add_argument("--sft-warmup-steps", type=int, default=60)
    pl.add_argument("--save-every", type=int, default=25)
    pl.add_argument("--audit-every", type=int, default=25)
    pl.add_argument("--wandb-project", default="pharmarl")
    pl.add_argument("--with-wandb", action="store_true",
                    help="Forward WANDB_API_KEY from your shell to the job")
    pl.add_argument("--dry-run", action="store_true",
                    help="Print the command without submitting")
    pl.set_defaults(func=cmd_launch)

    # status
    ps = sub.add_parser("status", help="list your H200 jobs")
    ps.set_defaults(func=cmd_status)

    # logs
    pg = sub.add_parser("logs", help="stream logs of a job")
    pg.add_argument("job_id")
    pg.set_defaults(func=cmd_logs)

    # cancel
    pc = sub.add_parser("cancel", help="cancel a running job")
    pc.add_argument("job_id")
    pc.set_defaults(func=cmd_cancel)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
