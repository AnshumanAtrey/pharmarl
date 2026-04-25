# Dockerfile for PharmaRL — deployable to HuggingFace Spaces (CPU runtime).
#
# Build context = repo root. Built image runs the FastAPI env server on port 8000.

FROM python:3.11-slim

# RDKit / TDC pull in heavy native deps; install build tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libxrender1 \
        libxext6 \
        libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps separately to leverage Docker layer caching
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Copy the env code
COPY models.py     /app/models.py
COPY client.py     /app/client.py
COPY __init__.py   /app/__init__.py
COPY server        /app/server

# HuggingFace Spaces sets $PORT
ENV PORT=8000
EXPOSE 8000

# Pre-warm TDC oracles so the first request isn't a 30s download stall
RUN python -c "from server.oracles import score_qed, score_sa, score_toxicity, score_mpro_docking; \
               smi='CC(=O)Oc1ccccc1C(=O)O'; \
               print('warmup:', score_qed(smi), score_sa(smi), score_mpro_docking(smi), score_toxicity(smi))" || true

CMD ["python", "-m", "server.app"]
