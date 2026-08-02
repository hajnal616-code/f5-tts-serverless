FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# ---------------------------------------------------------
# SYSTEM PACKAGES (libsndfile1 & ffmpeg for audio processing)
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
# PYTHON TOOLING & DEPENDENCIES
# ---------------------------------------------------------
RUN python3 -m pip install --upgrade pip setuptools wheel

RUN python3 -m pip install \
        runpod \
        requests \
        soundfile \
        scipy \
        vocos \
        "huggingface_hub>=0.24" \
        "transformers>=4.43.0,<4.47.0" \
        f5-tts

# ---------------------------------------------------------
# DOWNLOAD HUNGARIAN F5-TTS MODEL FILES (mp3pintyo/F5-TTS-Hun & Maxdorger29)
# ---------------------------------------------------------
RUN python3 -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='mp3pintyo/F5-TTS-Hun', filename='model_122000-hun.pt', local_dir='/workspace'); \
hf_hub_download(repo_id='Maxdorger29/f5-tts-hungarian', filename='model_last_final.safetensors', local_dir='/workspace'); \
hf_hub_download(repo_id='Maxdorger29/f5-tts-hungarian', filename='vocab.txt', local_dir='/workspace'); \
hf_hub_download(repo_id='Maxdorger29/f5-tts-hungarian', filename='config.json', local_dir='/workspace')" || true

# ---------------------------------------------------------
# PRE-CACHE VOCOS VOCODER WEIGHTS & DEFAULT REF AUDIO
# ---------------------------------------------------------
RUN python3 -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='charactr/vocos-mel-24khz', filename='config.yaml', local_dir='/workspace/.cache/huggingface/hub'); \
hf_hub_download(repo_id='charactr/vocos-mel-24khz', filename='pytorch_model.bin', local_dir='/workspace/.cache/huggingface/hub')" || true

RUN curl -L -o /workspace/default_ref.wav "https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/sample1.wav" || true

# ---------------------------------------------------------
# COPY HANDLER & START WORKER
# ---------------------------------------------------------
COPY handler.py /workspace/handler.py

CMD ["python3", "-u", "/workspace/handler.py"]
