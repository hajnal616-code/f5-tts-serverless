# Használj egy hivatalos PyTorch CUDA képet alapnak (ez már tartalmazza az illesztőket)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Telepítsük a szükséges rendszerfüggőségeket
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Munkakönyvtár beállítása
WORKDIR /workspace

# Python függőségek telepítése (explicit módon a PyTorch indexből)
RUN pip install --no-cache-dir \
    runpod \
    torch \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# F5-TTS kód másolása és telepítése
COPY . .
RUN pip install --no-cache-dir -e .

# A handler futtatása
CMD ["python3", "handler.py"]
