# PharmaRL training Space — Path C fallback
# Runs scripts/train_grpo.py on T4 Small hardware against a remote env URL.
# Pushes trained LoRA to HF Hub on completion. Minimal HTTP server keeps the
# Space alive (HF kills Spaces that don't expose a port).

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/data/hf_cache \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Install PyTorch first (CUDA-matched)
RUN pip install --upgrade pip && \
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Project deps — pinned for reproducibility on Spaces
RUN pip install \
        unsloth \
        "trl>=0.11.0" \
        "transformers>=4.40.0" \
        peft accelerate \
        "openenv-core" \
        selfies rdkit-pypi PyTDC \
        wandb \
        huggingface_hub \
        fastapi uvicorn requests

WORKDIR /app
COPY . /app

EXPOSE 7860

# Entry: launch the heartbeat HTTP server on :7860, kick off training in
# the background. The HTTP server is just to satisfy HF Spaces health check.
CMD ["bash", "-c", "python /app/run.py"]
