import base64
import io
import torch
import runpod
import soundfile as sf

from f5_tts.api import F5TTS

# Pre-load the Hungarian F5-TTS model when the worker container boot (Warm start)
print("Betöltés: Maxdorger29 Magyar F5-TTS modell...")
try:
    tts_model = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file="/workspace/model_last_final.safetensors",
        vocab_file="/workspace/vocab.txt",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_ema=True,
    )
    print("Sikeres magyar F5-TTS modell betöltés!")
except Exception as e:
    print(f"Hiba a modell betöltésekor indításkor: {e}")
    tts_model = None


def handler(event):
    global tts_model

    # Backup lazy load if global load failed
    if tts_model is None:
        try:
            tts_model = F5TTS(
                model="F5TTS_v1_Base",
                ckpt_file="/workspace/model_last_final.safetensors",
                vocab_file="/workspace/vocab.txt",
                device="cuda" if torch.cuda.is_available() else "cpu",
                use_ema=True,
            )
        except Exception as e:
            return {"status": "error", "error": f"Modell betöltési hiba: {str(e)}"}

    try:
        input_data = event.get("input", {})
        text = input_data.get("text", "")
        ref_audio_url = input_data.get("ref_audio_url", "")
        ref_text = input_data.get("ref_text", "")

        if not text or not text.strip():
            return {"status": "error", "error": "Hiányzó 'text' paraméter!"}

        # F5TTS.infer returns (wav, sample_rate, spect)
        with torch.no_grad():
            res = tts_model.infer(
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

        # Encode numpy wav array to WAV bytes in memory
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        audio_bytes = buf.getvalue()

        # Base64 encode for response
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return {
            "status": "success",
            "audio_file": audio_b64,
            "audio_base64": audio_b64
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }


runpod.serverless.start({"handler": handler})
