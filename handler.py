import sys
import base64
import io
import torch
import runpod
import soundfile as sf
import traceback

print("RunPod Serverless Worker indítása...", flush=True)

# Globális modell példány
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
            model_type="F5-TTS",
            ckpt_file="/workspace/model_last_final.safetensors",
            vocab_file="/workspace/vocab.txt",
            device=device,
            use_ema=True,
        )
        print("Modell sikeresen betöltve (F5-TTS)!", flush=True)
    except Exception as e1:
        print(f"Hiba a model_type='F5-TTS' betöltésekor: {e1}", flush=True)
        try:
            tts_model = F5TTS(
                ckpt_file="/workspace/model_last_final.safetensors",
                vocab_file="/workspace/vocab.txt",
                device=device,
                use_ema=True,
            )
            print("Modell sikeresen betöltve (alapértelmezett model_type)!", flush=True)
        except Exception as e2:
            print(f"Hiba a magyar F5-TTS modell betöltésekor: {e2}", flush=True)
            raise e2

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

        with torch.no_grad():
            res = model.infer(
                gen_text=text.strip(),
                ref_file=ref_audio_url if ref_audio_url else "",
                ref_text=ref_text if ref_text else "",
            )

            if isinstance(res, tuple):
                wav = res[0]
                sr = res[1] if len(res) > 1 else 24000
            else:
                wav = res
                sr = 24000

        # WAV konvertálása memóriában
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


# Modell előbetöltése a konténer indításakor
try:
    print("Modell előbetöltése indításkor...", flush=True)
    get_model()
except Exception as e:
    print(f"Figyelem: Az indításkori modell betöltés meghiúsult: {e}", flush=True)

# RunPod Serverless worker indítása
print("Calling runpod.serverless.start...", flush=True)
runpod.serverless.start({"handler": handler})
