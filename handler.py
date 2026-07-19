import os
import base64
import tempfile
import traceback
from urllib.parse import urlparse
from urllib.request import urlopen, Request

import torch
import torchaudio
import runpod

from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process,
)
from f5_tts.model import DiT

# ---------- Globális modell betöltés (cold start egyszer) ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[boot] device={DEVICE}, torch={torch.__version__}")

# F5-TTS alap konfiguráció (a hivatalos repo defaultjai)
F5_MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

print("[boot] loading vocoder…")
VOCODER = load_vocoder()

print("[boot] loading F5-TTS model…")
# A load_model automatikusan letölti a HuggingFace checkpointot ha nincs cache-elve
MODEL = load_model(DiT, F5_MODEL_CFG, ckpt_path="hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors")
print("[boot] ready.")


def _download_to_tmp(url: str) -> str:
    """Letölti a signed URL-t egy ideiglenes fájlba és visszaadja az útvonalát."""
    parsed = urlparse(url)
    suffix = os.path.splitext(parsed.path)[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="ref_")
    os.close(fd)
    req = Request(url, headers={"User-Agent": "f5-tts-worker/1.0"})
    with urlopen(req, timeout=30) as resp, open(path, "wb") as f:
        f.write(resp.read())
    return path


def _resolve_ref_audio(ref: str) -> str:
    """Elfogad helyi útvonalat vagy http(s) URL-t. Mindig helyi fájl útvonalat ad vissza."""
    if ref.startswith("http://") or ref.startswith("https://"):
        return _download_to_tmp(ref)
    if not os.path.exists(ref):
        raise FileNotFoundError(f"A referenciahang nem található: {ref}")
    return ref


def handler(job):
    try:
        job_input = job.get("input", {}) or {}

        # A kliensünk `gen_text` / `text` néven küldi
        text = job_input.get("gen_text") or job_input.get("text")
        # A kliensünk `ref_audio_url` / `ref_audio` néven küldi
        ref_audio = job_input.get("ref_audio_url") or job_input.get("ref_audio")
        ref_text = job_input.get("ref_text", "") or ""

        speed = float(job_input.get("speed", 1.0))
        nfe_step = int(job_input.get("nfe_step", 32))   # 16=gyorsabb, 32=alapérték, 64=jobb minőség
        cross_fade_duration = float(job_input.get("cross_fade_duration", 0.15))

        if not text:
            return {"error": "Hiányzó 'text' / 'gen_text' input."}
        if not ref_audio:
            return {"error": "Hiányzó 'ref_audio' / 'ref_audio_url' input."}

        local_ref = _resolve_ref_audio(ref_audio)

        # F5-TTS előfeldolgozás (opcionális auto-transcript ha ref_text üres)
        ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(local_ref, ref_text)

        # Inferencia
        wav, sr, _ = infer_process(
            ref_audio_proc,
            ref_text_proc,
            text,
            MODEL,
            VOCODER,
            speed=speed,
            nfe_step=nfe_step,
            cross_fade_duration=cross_fade_duration,
        )

        # Torch tensor → WAV bytes → base64
        import io
        wav_tensor = torch.from_numpy(wav).unsqueeze(0) if not torch.is_tensor(wav) else wav
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        buf = io.BytesIO()
        torchaudio.save(buf, wav_tensor.cpu(), sr, format="wav")
        wav_bytes = buf.getvalue()

        # MP3-ra konvertálás ha kérték (kisebb payload)
        fmt = (job_input.get("format") or job_input.get("output_format") or "mp3").lower()
        audio_bytes = wav_bytes
        content_type = "audio/wav"
        if fmt == "mp3":
            try:
                import subprocess
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin:
                    fin.write(wav_bytes)
                    wav_path = fin.name
                mp3_path = wav_path.replace(".wav", ".mp3")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", wav_path, "-b:a", "96k", mp3_path],
                    check=True, capture_output=True,
                )
                with open(mp3_path, "rb") as f:
                    audio_bytes = f.read()
                content_type = "audio/mpeg"
                os.unlink(wav_path); os.unlink(mp3_path)
            except Exception as e:
                print(f"[warn] mp3 fallback wav miatt: {e}")

        b64 = base64.b64encode(audio_bytes).decode("ascii")

        # Cleanup
        if local_ref != ref_audio and local_ref.startswith("/tmp/"):
            try: os.unlink(local_ref)
            except: pass

        return {
            "audio_base64": b64,
            "content_type": content_type,
            "sample_rate": sr,
            "duration_sec": wav_tensor.shape[-1] / sr,
        }

    except Exception as e:
        print("[error]", traceback.format_exc())
        return {"error": f"{type(e).__name__}: {str(e)}"}


# ---- RunPod serverless entrypoint ----
runpod.serverless.start({"handler": handler})
