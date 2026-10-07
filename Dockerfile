FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[dashboard]"

# Pre-download model weights during build (avoids ~5min wait on first run)
# The "-v6" HF repo holds the v6d weights (CRITICAL-dominant fine-tune, 75% CRITICAL
# recall on held-out eval). Pinned to the v6d commit so builds are reproducible.
# Keep MODEL_REVISION in sync with src/triage/model.py.
ARG HF_TOKEN=""
ARG MODEL_REVISION=c42df7c5c77c85a12b2d059fe252f2f6bf5fd886
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('marcelo-earth/LFM2.5-VL-450M-satellite-triage-v6', revision='${MODEL_REVISION}', token='${HF_TOKEN}' or None); \
snapshot_download('LiquidAI/LFM2.5-VL-450M', ignore_patterns=['*.safetensors'], token='${HF_TOKEN}' or None)"

# Copy source
COPY src/ src/

# Configure
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.dashboard.app:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
