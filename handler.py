import os
import io
import base64
import tempfile
import requests
import torch
import torchaudio
import runpod

from huggingface_hub import hf_hub_download
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process
from f5_tts.model import DiT

# ---------------------------------------------------------------------------
# Cold start: load model + vocoder ONCE per worker process.
# ---------------------------------------------------------------------------
print("[boot] downloading F5-TTS Base checkpoint...")
CKPT_PATH = hf_hub_download(
    repo_id="SWivid/F5-TTS",
    filename="F5TTS_Base/model_1200000.safetensors",
)
print(f"[boot] checkpoint: {CKPT_PATH}")

print("[boot] loading vocoder...")
VOCODER = load_vocoder()

print("[boot] loading DiT model...")
MODEL = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=CKPT_PATH,
)
print("[boot] ready.")


def _fetch_ref_audio(url_or_path: str) -> str:
    """Accept a URL (signed Supabase URL) or a local path. Return a local path."""
    if url_or_path.startswith(("http://", "https://")):
        r = requests.get(url_or_path, timeout=30)
        r.raise_for_status()
        suffix = ".wav"
        # Preserve extension if present in the URL path
        lower = url_or_path.split("?", 1)[0].lower()
        for ext in (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"):
            if lower.endswith(ext):
                suffix = ext
                break
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    if not os.path.exists(url_or_path):
        raise FileNotFoundError(f"Reference audio not found: {url_or_path}")
    return url_or_path


def _encode_audio(wav, sample_rate: int, fmt: str = "mp3") -> tuple[str, str]:
    """Encode waveform (numpy or tensor) into base64 audio bytes."""
    if not torch.is_tensor(wav):
        wav = torch.tensor(wav)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)  # (1, T)
    buf = io.BytesIO()
    fmt = (fmt or "mp3").lower()
    if fmt == "wav":
        torchaudio.save(buf, wav, sample_rate, format="wav")
        mime = "audio/wav"
    else:
        # mp3 requires ffmpeg backend (installed in the Dockerfile)
        torchaudio.save(buf, wav, sample_rate, format="mp3")
        mime = "audio/mpeg"
    return base64.b64encode(buf.getvalue()).decode("utf-8"), mime


def handler(job):
    job_input = job.get("input", {}) or {}

    gen_text = job_input.get("gen_text") or job_input.get("text")
    ref_src = job_input.get("ref_audio_url") or job_input.get("ref_audio")
    ref_text = job_input.get("ref_text") or ""
    speed = float(job_input.get("speed", 0.95))
    nfe_step = int(job_input.get("nfe_step", 32))
    out_fmt = (job_input.get("format") or job_input.get("output_format") or "mp3").lower()

    if not gen_text:
        return {"error": "Missing 'gen_text' (or 'text')."}
    if not ref_src:
        return {"error": "Missing 'ref_audio_url' (or 'ref_audio')."}

    tmp_ref = None
    try:
        ref_path = _fetch_ref_audio(ref_src)
        tmp_ref = ref_path if ref_src.startswith(("http://", "https://")) else None

        wav, sr, _ = infer_process(
            ref_path,
            ref_text,
            gen_text,
            MODEL,
            VOCODER,
            speed=speed,
            nfe_step=nfe_step,
        )

        audio_b64, mime = _encode_audio(wav, sr, out_fmt)
        return {
            "audio_base64": audio_b64,
            "mime": mime,
            "sample_rate": sr,
            "format": "wav" if out_fmt == "wav" else "mp3",
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        if tmp_ref and os.path.exists(tmp_ref):
            try:
                os.remove(tmp_ref)
            except Exception:
                pass


runpod.serverless.start({"handler": handler})
