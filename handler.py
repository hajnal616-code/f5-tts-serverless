import os
import io
import time
import base64
import tempfile
import urllib.request
import runpod

# Hugging Face magyar F5-TTS modell azonosító
MODEL_NAME = os.environ.get("TTS_MODEL_ID", "sarpba/F5-TTS_V1_hun_v2")

# Globális modell példány gyűjtő
global_tts_model = None

def get_model():
    global global_tts_model
    if global_tts_model is not None:
        return global_tts_model

    print(f"Modell betöltése folyamatban... ({MODEL_NAME})", flush=True)
    import torch
    from f5_tts.api import F5TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Használt eszköz: {device}", flush=True)

    # Előre letöltött / helyi fájlok keresése a /workspace könyvtárban
    workspace_dir = "/workspace"
    ckpt_candidates = [
        os.path.join(workspace_dir, "model_927900.safetensors"),
        os.path.join(workspace_dir, "model_927900.pt"),
        os.path.join(workspace_dir, "model_309300.safetensors"),
        os.path.join(workspace_dir, "model_309300.pt"),
        os.path.join(workspace_dir, "model_122000-hun.pt"),
        os.path.join(workspace_dir, "model_last_final.safetensors")
    ]
    vocab_path = os.path.join(workspace_dir, "vocab.txt")

    for ckpt in ckpt_candidates:
        if os.path.exists(ckpt):
            print(f"Helyi modell checkpoint megtalálva: {ckpt}", flush=True)
            try:
                tts_model = F5TTS(
                    ckpt_file=ckpt,
                    vocab_file=vocab_path if os.path.exists(vocab_path) else "",
                    device=device,
                    use_ema=True,
                )
                print(f"Modell ({ckpt}) sikeresen betöltve!", flush=True)
                global_tts_model = tts_model
                return global_tts_model
            except Exception as e:
                print(f"Figyelem: {ckpt} betöltési hiba: {e}", flush=True)

    # Dinamikus letöltés a Hugging Face Hub-ról ha a helyi nem érhető el
    try:
        print(f"Modellfájlok letöltése a Hugging Face Hub-ról ({MODEL_NAME})...", flush=True)
        from huggingface_hub import hf_hub_download

        downloaded_ckpt = None
        downloaded_vocab = None

        try:
            downloaded_vocab = hf_hub_download(repo_id=MODEL_NAME, filename="vocab.txt")
        except Exception as ve:
            print(f"Vocab letöltési figyelem: {ve}", flush=True)
            if os.path.exists(vocab_path):
                downloaded_vocab = vocab_path

        filenames_to_try = [
            "model_927900.safetensors",
            "model_927900.pt",
            "model_309300.safetensors",
            "model_309300.pt",
            "model_122000-hun.pt",
            "model_last_final.safetensors"
        ]

        for fname in filenames_to_try:
            try:
                downloaded_ckpt = hf_hub_download(repo_id=MODEL_NAME, filename=fname)
                print(f"Sikeres {fname} letöltés: {downloaded_ckpt}", flush=True)
                break
            except Exception as de:
                pass

        if not downloaded_ckpt:
            raise ValueError(f"Egyetlen ismert modellfájlt sem sikerült letölteni a {MODEL_NAME} repóból!")

        tts_model = F5TTS(
            ckpt_file=downloaded_ckpt,
            vocab_file=downloaded_vocab if downloaded_vocab else "",
            device=device,
            use_ema=True,
        )
        print(f"Modell ({MODEL_NAME}) sikeresen betöltve a letöltött fájlból!", flush=True)
        global_tts_model = tts_model
        return global_tts_model

    except Exception as e:
        print(f"Modell betöltési hiba a HF Hub-ról: {e}", flush=True)
        raise e


def handler(job):
    """
    RunPod serverless handler a magyar F5-TTS modellhez.
    """
    job_input = job.get("input", {})
    text = job_input.get("text", "")
    ref_audio_raw = job_input.get("ref_audio") or job_input.get("ref_audio_url")
    ref_text = job_input.get("ref_text", "")

    if not text:
        return {"error": "A 'text' mező megadása kötelező!"}

    try:
        model = get_model()

        temp_ref_file = None
        ref_audio_path = "/workspace/default_ref.wav"

        if ref_audio_raw:
            if ref_audio_raw.startswith("http://") or ref_audio_raw.startswith("https://"):
                try:
                    temp_ref = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    urllib.request.urlretrieve(ref_audio_raw, temp_ref.name)
                    temp_ref_file = temp_ref.name
                    ref_audio_path = temp_ref.name
                except Exception as dl_err:
                    print(f"Hiba a referenciang URL letöltésekor: {dl_err}", flush=True)
            elif ref_audio_raw.startswith("data:audio") or len(ref_audio_raw) > 200:
                try:
                    b64_data = ref_audio_raw
                    if "," in b64_data:
                        b64_data = b64_data.split(",")[1]
                    audio_bytes = base64.b64decode(b64_data)
                    temp_ref = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    temp_ref.write(audio_bytes)
                    temp_ref.close()
                    temp_ref_file = temp_ref.name
                    ref_audio_path = temp_ref.name
                except Exception as b64_err:
                    print(f"Hiba a Base64 referenciaszubjektum dekódolásakor: {b64_err}", flush=True)

        if not os.path.exists(ref_audio_path):
            import numpy as np
            import soundfile as sf
            dummy_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(dummy_wav.name, np.zeros(24000, dtype=np.float32), 24000)
            ref_audio_path = dummy_wav.name
            temp_ref_file = dummy_wav.name

        print(f"Inferencia indítása... Szöveg: '{text[:50]}...'", flush=True)

        output_wav_file, _ = model.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text if ref_text else "",
            gen_text=text,
            file_type="wav",
        )

        with open(output_wav_file, "rb") as f:
            audio_data = f.read()

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        if temp_ref_file and os.path.exists(temp_ref_file):
            try:
                os.remove(temp_ref_file)
            except Exception:
                pass

        return {
            "status": "success",
            "audio": f"data:audio/wav;base64,{audio_b64}",
            "format": "wav"
        }

    except Exception as err:
        print(f"Inferencia hiba történt: {err}", flush=True)
        import traceback
        traceback.print_exc()
        return {"error": f"RunPod worker hiba történt: {str(err)}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
