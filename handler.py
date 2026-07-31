import os
import base64
import requests
import runpod

# -----------------------------
#  MAGYAR F5 TTS MODELL OSZTÁLY
# -----------------------------

class HungarianF5TTS:
    def __init__(self):
        # A Modal endpoint URL-je (Invoke URL)
        self.modal_url = os.getenv("MODAL_TTS_URL")  # <-- ezt RunPod env-ben add meg
        self.api_key = os.getenv("MODAL_API_KEY")    # <-- ezt is RunPod env-ben add meg

        if not self.modal_url:
            raise ValueError("MODAL_TTS_URL nincs beállítva RunPod environment variables-ben.")
        if not self.api_key:
            raise ValueError("MODAL_API_KEY nincs beállítva RunPod environment variables-ben.")

    def synthesize(self, text: str) -> bytes:
        """
        Magyar F5 TTS hívás Modal endpointtal.
        Visszatér: raw audio bytes (wav).
        """
        payload = {"text": text}

        response = requests.post(
            self.modal_url,
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )

        if response.status_code != 200:
            raise Exception(f"Modal TTS error: {response.text}")

        # Modal endpoint raw WAV bytes-t ad vissza
        return response.content


# Globális modellpéldány (RunPod csak egyszer tölti be)
tts_model = HungarianF5TTS()


# -----------------------------
#  RUNPOD HANDLER
# -----------------------------

def handler(event):
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


# -----------------------------
#  RUNPOD SERVERLESS START
# -----------------------------

runpod.serverless.start({"handler": handler})
