FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace

# System deps — ffmpeg is required for torchaudio mp3 encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 1) Python deps
RUN pip install --upgrade pip && \
    pip install \
        runpod \
        requests \
        hf_transfer \
        "huggingface_hub>=0.24" && \
    pip install f5-tts

# 2) Sanity check the CUDA stack early — fail fast if wheels don't match
RUN python -c "import torch, torchaudio; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'torchaudio', torchaudio.__version__)"

# 3) Preload F5-TTS Base checkpoint + vocoder into the image (cuts cold start)
RUN python -c "\
from huggingface_hub import hf_hub_download; \
p = hf_hub_download(repo_id='SWivid/F5-TTS', filename='F5TTS_Base/model_1200000.safetensors'); \
print('checkpoint cached at', p); \
from f5_tts.infer.utils_infer import load_vocoder; \
load_vocoder(); \
print('vocoder cached')"

# 4) Handler
COPY handler.py /workspace/handler.py

CMD ["python", "-u", "handler.py"]
