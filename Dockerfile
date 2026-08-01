FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# ---------------------------------------------------------
# SYSTEM PACKAG
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# PYTHON TOOLING
# ---------------------------------------------------------
RUN python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------
# INSTALL PYTHON DEPENDENCIES
# ---------------------------------------------------------
RUN pip install \
        runpod \
        requests \
        soundfile \
        "huggingface_hub>=0.24" \
        f5-tts

# ---------------------------------------------------------
# DOWNLOAD MAXDORGER29 MODEL FILES
# ---------------------------------------------------------
RUN python - << 'EOF'
from huggingface_hub import hf_hub_download

# Main model weights
hf_hub_download(
    repo_id="Maxdorger29/f5-tts-hungarian",
    filename="model_last_final.safetensors",
    local_dir="/workspace"
)

# Vocabulary file
hf_hub_download(
    repo_id="Maxdorger29/f5-tts-hungarian",
    filename="vocab.txt",
    local_dir="/workspace"
)

# Config file
hf_hub_download(
    repo_id="Maxdorger29/f5-tts-hungarian",
    filename="config.json",
    local_dir="/workspace"
)
EOF

# ---------------------------------------------------------
# COPY HANDLER
# ---------------------------------------------------------
COPY handler.py /workspace/handler.py

# ---------------------------------------------------------
# START SERVERLESS HANDLER
# ---------------------------------------------------------
CMD ["python", "/workspace/handler.py"]
