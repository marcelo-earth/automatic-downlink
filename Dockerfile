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
ARG HF_TOKEN=""
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('marcelo-earth/LFM2.5-VL-450M-satellite-triage', token='${HF_TOKEN}' or None); \
snapshot_download('LiquidAI/LFM2.5-VL-450M', ignore_patterns=['*.safetensors'], token='${HF_TOKEN}' or None)"

# Copy source
COPY src/ src/

# Configure
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.dashboard.app:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
