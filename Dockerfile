FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# ------------------------------------------------------
# SYSTEM PACKAGES
# --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# PYTHON TOOLING
# ---------------------------------------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel

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
# DOWNLOAD MAXDORGER29 HUNGARIAN MODEL FILES
# ---------------------------------------------------------
RUN python3 -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='Maxdorger29/f5-tts-hungarian', filename='model_last_final.safetensors', local_dir='/workspace'); \
hf_hub_download(repo_id='Maxdorger29/f5-tts-hungarian', filename='vocab.txt', local_dir='/workspace'); \
hf_hub_download(repo_id='Maxdorger29/f5-tts-hungarian', filename='config.json', local_dir='/workspace')"

# ---------------------------------------------------------
# PRE-CACHE BASE VOCODER WEIGHTS (CHARACTR/VOCOS-MEL-24KHZ)
# ---------------------------------------------------------
RUN python3 -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='charactr/vocos-mel-24khz', filename='config.yaml'); \
hf_hub_download(repo_id='charactr/vocos-mel-24khz', filename='pytorch_model.bin')" || true

# ---------------------------------------------------------
# COPY HANDLER
# ---------------------------------------------------------
COPY handler.py /workspace/handler.py

# ---------------------------------------------------------
# START SERVERLESS HANDLER
# ---------------------------------------------------------
CMD ["python3", "/workspace/handler.py"]
