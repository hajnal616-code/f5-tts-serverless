# =========================================================
# RUNPOD SERVERLESS DOCKERFILE FOR F5-TTS HUNGARIAN WORKER
# =========================================================

# 1. Base Image: RunPod PyTorch 2.4.0 with CUDA 12.4.1 support
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# ---------------------------------------------------------
# SYSTEM DEPENDENCIES (ffmpeg, libsndfile, git, curl, etc.)
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        ca-certificates \
        libsndfile1 \
        libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# PYTHON PIP & BASE PACKAGES
# ---------------------------------------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel

# Re-install/align torch and torchaudio to guarantee matching C++ ABI symbols (torch/torchaudio 2.4.1+cu124)
RUN python3 -m pip install --no-cache-dir "torch==2.4.1+cu124" "torchaudio==2.4.1+cu124" --index-url https://download.pytorch.org/whl/cu124

RUN python3 -m pip install \
        runpod \
        requests \
        soundfile \
        scipy \
        vocos \
        pypinyin \
        jieba \
        librosa \
        ema-pytorch \
        cached-path \
        hydra-core \
        omegaconf \
        matplotlib \
        tqdm \
        einops \
        pydub \
        tomli \
        pydantic \
        "huggingface_hub>=0.24" \
        "transformers>=4.43.0,<4.47.0"

RUN python3 -m pip install --no-deps f5-tts

# ---------------------------------------------------------
# DOWNLOAD SARPBA/F5-TTS_V1_HUN_V2 HUNGARIAN F5-TTS MODEL FILES
# ---------------------------------------------------------
RUN python3 -c "\
from huggingface_hub import hf_hub_download;\
print('Downloading vocab.txt...');\
hf_hub_download(repo_id='sarpba/F5-TTS_V1_hun_v2', filename='vocab.txt', local_dir='/workspace/checkpoints/hungarian');\
print('Downloading model checkpoint...');\
try:\
    hf_hub_download(repo_id='sarpba/F5-TTS_V1_hun_v2', filename='model_927900.safetensors', local_dir='/workspace/checkpoints/hungarian');\
except Exception as e:\
    print('Failed model_927900.safetensors, trying model_122000-hun.pt...', e);\
    hf_hub_download(repo_id='sarpba/F5-TTS_V1_hun_v2', filename='model_122000-hun.pt', local_dir='/workspace/checkpoints/hungarian');\
print('Hungarian F5-TTS Model Files Downloaded Successfully!')\
"

# Copy handler script
COPY handler.py /workspace/handler.py

# Set default execution command for RunPod Serverless container
CMD [ "python3", "-u", "/workspace/handler.py" ]
