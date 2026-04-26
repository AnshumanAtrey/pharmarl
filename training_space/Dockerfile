# PharmaRL training Space — H200 GPU run (issue #14, parallel insurance)
# Runs scripts/train_grpo.py on Hopper hardware against a remote env URL.
# Pushes trained LoRA to HF Hub on completion. Minimal HTTP server keeps the
# Space alive (HF kills Spaces that don't expose a port).
#
# Stack pinned for sm_90 (H200) compatibility:
#   - PyTorch 2.6.0 + CUDA 12.4 (Hopper-aware)
#   - bitsandbytes >= 0.45.5 (LLM.int8 on Hopper)
#   - xformers >= 0.0.30 (sm_90 kernels; older 0.0.27 won't build/run)
#   - unsloth 2025.2.15 + trl 0.13.0 + transformers 4.46.3 (matched triplet)

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/data/hf_cache \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Hopper-compatible kernels — install before unsloth so its resolver picks them up
RUN pip install \
        "bitsandbytes>=0.45.5" \
        "xformers>=0.0.30"

# Pinned training stack for H200 (per issue #14 acceptance criteria)
RUN pip install \
        "transformers==4.46.3" \
        "trl==0.13.0" \
        "unsloth==2025.2.15" \
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
