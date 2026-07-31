import os
import requests

class HungarianF5TTS:
    def __init__(self):
        self.modal_url = "https://api.modal.com/api/v1/endpoints/YOUR_ENDPOINT_ID"   # <-- ezt megadod
        self.api_key = os.getenv("MODAL_API_KEY")  # RunPod env variable

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
