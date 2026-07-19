# CUDA 12.1 + cuDNN, Ubuntu 22.04 — F5-TTS-hez ajánlott
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch

# Rendszer csomagok
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3-pip python3.10-venv \
      git ffmpeg libsndfile1 curl ca-certificates \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1) PyTorch CUDA 12.1 build
RUN pip install --upgrade pip && \
    pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 2) F5-TTS + runtime függőségek
RUN pip install \
      f5-tts \
      runpod \
      soundfile \
      numpy \
      transformers \
      accelerate

# 3) Modell előmelegítés build közben (opcionális, de gyorsítja a cold startot)
#    Ha nem akarod a nagy image-et, kommenteld ki — futáskor tölti le.
RUN python -c "\
from f5_tts.infer.utils_infer import load_vocoder, load_model; \
from f5_tts.model import DiT; \
load_vocoder(); \
load_model(DiT, dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4), \
           ckpt_path='hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors'); \
print('models cached')"

# 4) Build-time sanity check
RUN python -c "import torch, torchaudio; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ta', torchaudio.__version__)"

# 5) Handler
COPY handler.py /workspace/handler.py

CMD ["python", "-u", "/workspace/handler.py"]
