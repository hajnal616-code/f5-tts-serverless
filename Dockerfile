# Base image standard PyTorch with CUDA 12.1 and Python 3.10
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# ---------------------------------------------------------
# SYSTEM DEPENDENCIES (FFmpeg, git, build essentials)
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# PYTHON DEPENDENCIES & F5-TTS INSTALLATION
# ---------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir \
    runpod \
    soundfile \
    requests \
    numpy \
    torchvision \
    torchaudio \
    huggingface_hub \
    vocos \
    f5-tts

# ---------------------------------------------------------
# DOWNLOAD SARPBA/F5-TTS_V1_HUN_V2 HUNGARIAN F5-TTS MODEL FILES
# ---------------------------------------------------------
ENV TTS_MODEL_ID="sarpba/F5-TTS_V1_hun_v2"

RUN python3 -c "from huggingface_hub import hf_hub_download; \
hf_hub_download(repo_id='sarpba/F5-TTS_V1_hun_v2', filename='model_927900.safetensors', local_dir='/workspace'); \
hf_hub_download(repo_id='sarpba/F5-TTS_V1_hun_v2', filename='model_927900.pt', local_dir='/workspace'); \
hf_hub_download(repo_id='sarpba/F5-TTS_V1_hun_v2', filename='vocab.txt', local_dir='/workspace'); \
hf_hub_download(repo_id='sarpba/F5-TTS_V1_hun_v2', filename='setting.json', local_dir='/workspace')" || true


# ---------------------------------------------------------
# PRE-CACHE VOCOS VOCODER WEIGHTS & DEFAULT REF AUDIO
# ---------------------------------------------------------
RUN python3 -c "from vocos import Vocos; Vocos.from_pretrained('charactr/vocos-mel-24khz')" || true

# Download default reference audio sample
RUN wget -q -O /workspace/default_ref.wav "https://raw.githubusercontent.com/SWivid/F5-TTS/main/tests/infer/examples/basic/basic_ref_en.wav" || true

# Copy worker handler script
COPY handler.py /workspace/handler.py

# Launch RunPod worker handler
CMD ["python3", "-u", "/workspace/handler.py"]
