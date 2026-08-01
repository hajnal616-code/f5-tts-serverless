import base64
import io
import torch
import runpod
import soundfile as sf

from f5_tts.api import F5TTS

# Lazy-init modellpéldánya
tts_model = None


class HungarianF5TTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # A három fájl a Hugging Face-ből
        self.model = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file="/workspace/model_last_final.safetensors",
            vocab_file="/workspace/vocab.txt",
            device=self.device,
            use_ema=True,
        )

    def synthesize(self, text: str) -> bytes:
        """
        Magyar F5 TTS → wav → raw bytes
        """
        with torch.no_grad():
            wav = self.model.inference(text)  # numpy array [samples]

        # WAV-ba írás memóriába
        buf = io.BytesIO()
        sf.write(buf, wav, 24000, format="WAV")
        return buf.getvalue()


def handler(event):
    global tts_model

    # Lazy-init: csak akkor töltjük be a modellt, amikor már biztosan létezik minden fájl
    if tts_model is None:
        tts_model = HungarianF5TTS()

    try:
        text = event["input"]["text"]

        # Magyar TTS → audio bytes
        audio_bytes = tts_model.synthesize(text)

        # Base64 kódolás JSON válaszhoz
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "status": "success",
            "audio_base64": audio_b64
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


runpod.serverless.start({"handler": handler})
