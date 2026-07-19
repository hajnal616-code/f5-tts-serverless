# Használjuk a PyTorch hivatalos képét, ami már tartalmazza a CUDA-t és a torch-ot
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Telepítsük a szükséges rendszerfüggőségeket
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Csak azt telepítsük, ami nincs benne az alap képben
# A 'torch' és 'torchaudio' már benne van az alap képben!
RUN pip install --no-cache-dir runpod

# F5-TTS kód másolása és telepítése
COPY . .
RUN pip install --no-cache-dir -e .

# A handler futtatása
CMD ["python3", "handler.py"]
