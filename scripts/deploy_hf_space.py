"""Deploy PharmaRL to HuggingFace Spaces (Docker, CPU free tier).

Reads HF_TOKEN, HF_USERNAME, HF_SPACE_NAME from .env (gitignored).
Idempotent — creates the Space if missing, then uploads the repo contents.

Usage:
    python scripts/deploy_hf_space.py            # full deploy
    python scripts/deploy_hf_space.py --dry-run  # show what would happen
    python scripts/deploy_hf_space.py --restart  # restart the existing Space (no upload)

Notes:
- CPU free tier — no GPU, no credit consumption from runtime.
- Files matching .gitignore patterns are skipped (research/, resources/, .env, oracle/, etc.).
- HF Spaces builds the Docker image on their infra (linux/amd64). We don't build locally.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> dict[str, str]:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        print(f"[FAIL] {env_path} missing — copy .env.example and fill in HF_TOKEN.", file=sys.stderr)
        sys.exit(2)
    out: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
        os.environ.setdefault(k.strip(), v.strip())
    for required in ("HF_TOKEN", "HF_USERNAME", "HF_SPACE_NAME"):
        if not out.get(required):
            print(f"[FAIL] {required} missing from .env", file=sys.stderr)
            sys.exit(2)
    return out


# Files we explicitly EXCLUDE from the upload — keep the Space lean and safe.
# Most exclusions are already covered by .gitignore but we double-check
# because HfApi.upload_folder doesn't always honor .gitignore by default.
_EXCLUDE_PATTERNS = [
    ".env",
    ".env.local",
    ".env.*.local",
    ".venv/**",
    ".git/**",
    "__pycache__/**",
    "**/__pycache__/**",
    "*.py[cod]",
    "research/**",
    "resources/**",
    "oracle/**",         # 70MB+ of TDC model weights — Space rebuilds them
    "docking/**",
    "*.pdb",
    "*.pdbqt",
    "*.pkl",
    "*.pt",
    "*.bin",
    "*.safetensors",
    "wandb/**",
    "mlruns/**",
    ".pytest_cache/**",
    ".mypy_cache/**",
    ".ruff_cache/**",
    "*.log",
    ".DS_Store",
    ".coverage",
    "htmlcov/**",
    "/tmp/**",
    "tests/**",          # not needed in production Space
    "examples/**",       # devspace, not Space artifact
    "research/**",
    "scripts/smoke_notebook_locally.py",  # local-only debug
    "scripts/deploy_hf_space.py",         # don't ship the deploy script itself
    "colab/**",          # notebook is for Colab, not Space
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen, don't upload.")
    parser.add_argument("--restart", action="store_true", help="Restart the Space without re-uploading.")
    parser.add_argument("--private", action="store_true", help="Create the Space as private (default: public).")
    args = parser.parse_args(argv)

    env = load_env()
    repo_id = f"{env['HF_USERNAME']}/{env['HF_SPACE_NAME']}"

    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi(token=env["HF_TOKEN"])
    me = api.whoami()
    print(f"[OK]   auth: {me.get('name')} ({me.get('type')})")

    # Check if Space exists
    try:
        info = api.space_info(repo_id)
        exists = True
        print(f"[OK]   Space {repo_id} exists (sdk={info.sdk}).")
    except RepositoryNotFoundError:
        exists = False
        print(f"[NEW]  Space {repo_id} does not exist — will create.")

    if args.restart:
        if not exists:
            print("[FAIL] --restart requires the Space to already exist.", file=sys.stderr)
            return 2
        if args.dry_run:
            print(f"[DRY]  Would restart Space {repo_id}")
            return 0
        api.restart_space(repo_id)
        print(f"[OK]   Restarted {repo_id}")
        return 0

    if args.dry_run:
        print()
        print(f"[DRY]  Would create Space (if missing): {repo_id}")
        print(f"[DRY]  Would upload repo from: {REPO_ROOT}")
        print(f"[DRY]  Would skip {len(_EXCLUDE_PATTERNS)} ignore patterns including:")
        for p in _EXCLUDE_PATTERNS[:10]:
            print(f"          {p}")
        print(f"[DRY]  ... etc.")
        # Show top-level files that would actually be uploaded
        from pathlib import Path
        candidates = sorted(p for p in REPO_ROOT.iterdir() if not p.name.startswith("."))
        print()
        print(f"[DRY]  Top-level entries the Space would receive:")
        for p in candidates:
            kind = "dir" if p.is_dir() else "file"
            size = "" if p.is_dir() else f" ({p.stat().st_size}B)"
            print(f"          {kind:4s} {p.name}{size}")
        return 0

    # Create the Space if missing
    if not exists:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )
        print(f"[OK]   Created Space {repo_id} (sdk=docker, private={args.private})")

    # Upload the repo
    print(f"[...]  Uploading {REPO_ROOT} → {repo_id} (this can take a few minutes)")
    api.upload_folder(
        folder_path=str(REPO_ROOT),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=_EXCLUDE_PATTERNS,
        commit_message="Deploy PharmaRL multi-target env (DRD2 + GSK3B + JNK3)",
    )
    print(f"[OK]   Upload complete.")
    print()
    print(f"  Space URL:    https://huggingface.co/spaces/{repo_id}")
    print(f"  Direct API:   https://{env['HF_USERNAME']}-{env['HF_SPACE_NAME']}.hf.space")
    print()
    print(f"  HF will now build the Docker image. Watch progress at:")
    print(f"    https://huggingface.co/spaces/{repo_id}?logs=build")
    print()
    print(f"  First build: ~5-10 min (compiles native deps + downloads TDC oracles).")
    print(f"  Once 'Running': curl https://{env['HF_USERNAME']}-{env['HF_SPACE_NAME']}.hf.space/health")
    return 0


if __name__ == "__main__":
    sys.exit(main())
