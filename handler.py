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

MODEL_NAME = "mp3pintyo/F5-TTS-Hun"
DEFAULT_REF_FILE = "/workspace/default_ref.wav"
DEFAULT_REF_TEXT = "Magyar teszt referencia hang."

def ensure_default_ref_audio():
    if os.path.exists(DEFAULT_REF_FILE) and os.path.getsize(DEFAULT_REF_FILE) > 500:
        return DEFAULT_REF_FILE

    sample_urls = [
        "https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/sample1.wav",
        "https://raw.githubusercontent.com/SWivid/F5-TTS/main/tests/infer/examples/basic/basic_ref_en.wav"
    ]
    for url in sample_urls:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200 and len(r.content) > 500:
                with open(DEFAULT_REF_FILE, "wb") as f:
                    f.write(r.content)
                print(f"Alapértelmezett referencia hang letöltve: {url}", flush=True)
                return DEFAULT_REF_FILE
        except Exception as e:
            print(f"Nem sikerült letölteni letöltési URL-ről: {url} ({e})", flush=True)

    # Fallback: Create synthetic 2-second WAV
    try:
        import numpy as np
        sr = 24000
        t = np.linspace(0, 2, sr * 2, endpoint=False)
        audio = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        sf.write(DEFAULT_REF_FILE, audio, sr, format="WAV")
        print("Szintetikus referencia hang létrehozva.", flush=True)
    except Exception as e:
        print(f"Hiba a szintetikus hang létrehozásakor: {e}", flush=True)

    return DEFAULT_REF_FILE


# Global model instance
tts_model = None

def get_model():
    global tts_model
    if tts_model is not None:
        return tts_model

    print(f"Magyar F5-TTS modell betöltése ({MODEL_NAME}) CUDA/CPU eszközre...", flush=True)
    from f5_tts.api import F5TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Használt eszköz: {device}", flush=True)

    # Priority 1: Check mp3pintyo checkpoint file
    pt_path = "/workspace/model_122000-hun.pt"
    vocab_path = "/workspace/vocab.txt"
    safetensors_path = "/workspace/model_last_final.safetensors"

    if os.path.exists(pt_path):
        try:
            print(f"Próbálkozás mp3pintyo/F5-TTS-Hun betöltésével ({pt_path})...", flush=True)
            tts_model = F5TTS(
                ckpt_file=pt_path,
                vocab_file=vocab_path if os.path.exists(vocab_path) else "",
                device=device,
                use_ema=True,
            )
            print("mp3pintyo/F5-TTS-Hun modell sikeresen betöltve!", flush=True)
            return tts_model
        except Exception as e:
            print(f"Figyelem: pt modell fájl betöltési hiba: {e}", flush=True)

    # Priority 2: Safetensors model fallback
    if os.path.exists(safetensors_path):
        try:
            print(f"Próbálkozás F5-TTS safetensors modellel ({safetensors_path})...", flush=True)
            tts_model = F5TTS(
                ckpt_file=safetensors_path,
                vocab_file=vocab_path if os.path.exists(vocab_path) else "",
                device=device,
                use_ema=True,
            )
            print("F5-TTS safetensors modell sikeresen betöltve!", flush=True)
            return tts_model
        except Exception as e:
            print(f"Figyelem: safetensors modell betöltési hiba: {e}", flush=True)

    # Priority 3: Direct repo load
    try:
        print(f"Próbálkozás közvetlen {MODEL_NAME} hf repo betöltéssel...", flush=True)
        tts_model = F5TTS(
            model_name=MODEL_NAME,
            device=device,
        )
        print("Modell sikeresen betöltve!", flush=True)
        return tts_model
    except Exception as e:
        print(f"Modell betöltési hiba: {e}", flush=True)
        raise e


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
        ref_audio_base64 = input_data.get("ref_audio_base64", "")
        ref_audio_url = input_data.get("ref_audio_url", "")
        ref_text_in = input_data.get("ref_text", "")

        if not text or not str(text).strip():
            return {"status": "error", "error": "Hiányzó vagy üres 'text' paraméter!"}

        gen_text = str(text).strip()
        print(f"Generálás indítása szövegre: '{gen_text[:50]}...'", flush=True)

        # Handle reference audio decoding
        ref_file = None
        tmp_path = "/tmp/ref_audio.wav"

        # 1. Check ref_audio_base64 input payload parameter
        if ref_audio_base64 and str(ref_audio_base64).strip():
            try:
                b64_str = str(ref_audio_base64).strip()
                if "," in b64_str:
                    b64_str = b64_str.split(",")[-1]
                audio_bytes = base64.b64decode(b64_str)
                with open(tmp_path, "wb") as f:
                    f.write(audio_bytes)
                ref_file = tmp_path
                print(f"Base64 referencia hang dekódolva /tmp/ref_audio.wav ({len(audio_bytes)} bájt)", flush=True)
            except Exception as b64_err:
                print(f"Hiba a ref_audio_base64 dekódolásakor: {b64_err}", flush=True)

        # 2. Check ref_audio_url fallback parameter if ref_audio_base64 wasn't provided or failed
        if not ref_file and ref_audio_url and str(ref_audio_url).strip():
            ref_url = str(ref_audio_url).strip()
            if ref_url.startswith("http://") or ref_url.startswith("https://"):
                try:
                    print(f"Referencia hang letöltése URL-ről: {ref_url}", flush=True)
                    r = requests.get(ref_url, timeout=15)
                    r.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        f.write(r.content)
                    ref_file = tmp_path
                except Exception as dl_err:
                    print(f"Hiba a referencia hang letöltésekor: {dl_err}", flush=True)
            elif ref_url.startswith("data:audio") or len(ref_url) > 256:
                try:
                    b64_data = ref_url.split(",")[-1]
                    audio_data = base64.b64decode(b64_data)
                    with open(tmp_path, "wb") as f:
                        f.write(audio_data)
                    ref_file = tmp_path
                except Exception as b64_err:
                    print(f"Hiba a base64 URL dekódolásakor: {b64_err}", flush=True)
            elif os.path.exists(ref_url):
                ref_file = ref_url

        # 3. Fallback to default reference audio if none provided
        if not ref_file or not os.path.exists(ref_file):
            print("Alapértelmezett referencia hang használata...", flush=True)
            ref_file = ensure_default_ref_audio()

        # Handle reference text
        ref_text = DEFAULT_REF_TEXT
        if ref_text_in and str(ref_text_in).strip():
            ref_text = str(ref_text_in).strip()

        print(f"F5TTS.infer hívás: ref_file='{ref_file}', ref_text='{ref_text[:40]}...', gen_text='{gen_text[:30]}...'", flush=True)

        with torch.no_grad():
            res = model.infer(
                ref_file=ref_file,
                ref_text=ref_text,
                gen_text=gen_text,
            )

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
            "audio_base64": audio_b64,
            "sample_rate": sr
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
    ensure_default_ref_audio()
except Exception as e:
    print(f"Figyelem: Az indításkori modell betöltés meghiúsult: {e}", flush=True)

# Start the RunPod Serverless worker
print("Calling runpod.serverless.start...", flush=True)
runpod.serverless.start({"handler": handler})
