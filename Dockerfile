FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python tooling
RUN python -m pip install --upgrade pip setuptools wheel

# IMPORTANT:
# transformers 5.x is incompatible with torch 2.4.x in this image because it imports DTensor.
# Pin transformers/tokenizers to stable 4.x versions.
RUN cat > /tmp/constraints.txt <<'EOF'
transformers==4.46.3
tokenizers==0.20.3
numpy==1.26.4
tomlkit==0.15.1
EOF

# Install runtime + F5-TTS with pinned dependency constraints
RUN pip install -c /tmp/constraints.txt \
        runpod \
        requests \
        hf_transfer \
        "huggingface_hub>=0.24" \
        f5-tts \
    && pip install -c /tmp/constraints.txt --force-reinstall \
        "transformers==4.46.3" \
        "tokenizers==0.20.3" \
        "numpy==1.26.4" \
        "tomlkit==0.15.1"

# Quick dependency check
RUN python -c "import torch, torchaudio; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchaudio', torchaudio.__version__)" && \
    python -c "import transformers; print('transformers', transformers.__version__)"

# Pre-cache F5-TTS checkpoint.
# Do NOT call load_vocoder() here; importing full F5 inference during build was where the failure happened.
RUN python -c "\
from huggingface_hub import hf_hub_download; \
p = hf_hub_download(repo_id='SWivid/F5-TTS', filename='F5TTS_Base/model_1200000.safetensors'); \
print('F5 checkpoint cached at', p)"

# Optional: cache common vocoder repo files without importing F5 runtime.
RUN python -c "\
from huggingface_hub import snapshot_download; \
p = snapshot_download(repo_id='charactr/vocos-mel-24khz'); \
print('vocoder cached at', p)"

COPY handler.py /workspace/handler.py

CMD ["python", "/workspace/handler.py"]
