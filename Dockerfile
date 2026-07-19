# PyTorch runtime CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Rendszerfüggőségek
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 1) Takarítás: ne keveredjen a conda/pip torch stack
# (A base image gyakran tartalmaz torch-ot conda-val.)
RUN pip uninstall -y torch torchaudio torchvision torchtext || true && \
    conda remove -y pytorch torchaudio torchvision torchtext || true && \
    conda clean -a -y

# 2) Telepítsünk egy összepasszoló PyTorch + torchaudio + torchvision triót ugyanabból a csatornából
# CUDA 12.1 wheel-ek a PyTorch hivatalos indexéről
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.3.0 \
      torchvision==0.18.0 \
      torchaudio==2.3.0

# 3) Telepítsük a runpodot és az F5-TTS-t
RUN pip install runpod && \
    pip install git+https://github.com/SWivid/F5-TTS.git

# 4) Gyors ellenőrzés build közben (ha itt elhasal, rögtön látod)
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" && \
    python -c "import torchaudio; print('torchaudio', torchaudio.__version__)"

# Handler
COPY handler.py /workspace/handler.py

CMD ["python", "handler.py"]

