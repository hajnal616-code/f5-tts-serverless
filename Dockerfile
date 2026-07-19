# Használjuk a PyTorch hivatalos képét
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Telepítsük a szükséges rendszerfüggőségeket
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Telepítsük a futtatáshoz szükséges alap csomagokat
# A f5-tts modellt itt közvetlenül a GitHubról telepítjük
RUN pip install --no-cache-dir runpod \
    && pip install --no-cache-dir git+https://github.com/SWivid/F5-TTS.git

# A handler fájlt másoljuk be
COPY handler.py .

# A handler futtatása
CMD ["python3", "handler.py"]
