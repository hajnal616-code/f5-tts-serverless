import sys
import os
import base64
import io
import torch
import runpod
import soundfile as sf
import traceback
import requests

print("RunPod Serverless Worker indítása...", flush=True)

# Global model instance
tts_model = None

def get_model():
    global tts_model
    if tts_model is not None:
        return tts_model

    print("Maxdorger29 Magyar F5-TTS modell betöltése CUDA/CPU eszközre...", flush=True)
    from f5_tts.api import F5TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Használt eszköz: {device}", flush=True)

    try:
        tts_model = F5TTS(
            ckpt_file="/workspace/model_last_final.safetensors",
            vocab_file="/workspace/vocab.txt",
            device=device,
            use_ema=True,
        )
        print("Modell sikeresen betöltve!", flush=True)
    except Exception as e1:
        print(f"Hiba a modell betöltésekor: {e1}", flush=True)
        raise e1

    return tts_model


def handler(event):
    print(f"Beérkező kérés feldolgozása: {event}", flush=True)
    
    try:
        model = get_model()
    except Exception as e:
        err_msg = f"Modell inicializálási hiba: {str(e)}\n{traceback.format_exc()}"
        print(err_msg, flush=True)
        return {"status": "error", "error": err_msg}

    try:
        input_data = event.get("input", {})
        text = input_data.get("text", "")
        ref_audio_url = input_data.get("ref_audio_url", "")
        ref_text = input_data.get("ref_text", "")

        if not text or not text.strip():
            return {"status": "error", "error": "Hiányzó vagy üres 'text' paraméter!"}

        print(f"Generálás indítása szövegre: '{text[:50]}...'", flush=True)

        # Handle optional reference audio
        ref_file = None
        if ref_audio_url and str(ref_audio_url).strip():
            ref_url = str(ref_audio_url).strip()
            if ref_url.startswith("http://") or ref_url.startswith("https://"):
                try:
                    print(f"Referencia hang letöltése: {ref_url}", flush=True)
                    r = requests.get(ref_url, timeout=15)
                    r.raise_for_status()
                    tmp_path = "/tmp/ref_audio.wav"
                    with open(tmp_path, "wb") as f:
                        f.write(r.content)
                    ref_file = tmp_path
                except Exception as dl_err:
                    print(f"Hiba a referencia hang letöltésekor: {dl_err}", flush=True)
            elif ref_url.startswith("data:audio") or len(ref_url) > 256:
                try:
                    b64_data = ref_url.split(",")[-1]
                    audio_data = base64.b64decode(b64_data)
                    tmp_path = "/tmp/ref_audio.wav"
                    with open(tmp_path, "wb") as f:
                        f.write(audio_data)
                    ref_file = tmp_path
                except Exception as b64_err:
                    print(f"Hiba a base64 referencia hang dekódolásakor: {b64_err}", flush=True)
            elif os.path.exists(ref_url):
                ref_file = ref_url

        infer_kwargs = {
            "gen_text": text.strip(),
            "ref_file": ref_file,  # None if no reference audio provided
        }
        if ref_text and str(ref_text).strip():
            infer_kwargs["ref_text"] = str(ref_text).strip()

        with torch.no_grad():
            res = model.infer(**infer_kwargs)

            if isinstance(res, tuple):
                wav = res[0]
                sr = res[1] if len(res) > 1 else 24000
            else:
                wav = res
                sr = 24000

        # Encode numpy wav array to WAV bytes in memory
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        audio_bytes = buf.getvalue()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        print("Sikeres beszédgenerálás!", flush=True)

        return {
            "status": "success",
            "audio_file": audio_b64,
            "audio_base64": audio_b64
        }

    except Exception as e:
        err_str = f"Hiba a generálás során: {str(e)}\n{traceback.format_exc()}"
        print(err_str, flush=True)
        return {
            "status": "error",
            "error": err_str
        }


# Warm load model on worker container boot
try:
    print("Modell előbetöltése indításkor...", flush=True)
    get_model()
except Exception as e:
    print(f"Figyelem: Az indításkori modell betöltés meghiúsult: {e}", flush=True)

# Start the RunPod Serverless worker
print("Calling runpod.serverless.start...", flush=True)
runpod.serverless.start({"handler": handler})
