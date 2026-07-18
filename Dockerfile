FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN git clone https://github.com/SWivid/F5-TTS.git
WORKDIR /workspace/F5-TTS
RUN pip install --no-cache-dir -e .

COPY handler.py .

CMD ["python3", "-m", "src.f5_tts.infer.infer_gradio", "--host", "0.0.0.0", "--port", "7860"]