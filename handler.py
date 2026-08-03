import sys
import os
import re
import base64
import io
import torch
import runpod
import soundfile as sf
import traceback
import requests

print("RunPod Serverless Worker indítása...", flush=True)

MODEL_NAME = os.environ.get("TTS_MODEL_ID", "sarpba/F5-TTS_V1_hun_v2")
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")

DEFAULT_REF_FILE = "/workspace/default_ref.wav"
DEFAULT_REF_TEXT = "Magyar teszt referencia hang."

def normalize_hungarian_text(text: str) -> str:
    """
    Magyar nyelvű szövegnormalizáló:
    - Számok, dátumok, pénznemek (Ft), mértékegységek (km, kg) és mozaikszavak átírása fonetikus szöveggé.
    """
    if not text or not text.strip():
        return ""

    t = text.strip()

    t = re.sub(r'\bFt\b|\bft\b', ' forint', t)
    t = re.sub(r'\bkm\b|\bKM\b', ' kilométer', t)
    t = re.sub(r'\bkg\b|\bKG\b', ' kilogramm', t)
    t = re.sub(r'\bcm\b', ' centiméter', t)
    t = re.sub(r'\bmm\b', ' milliméter', t)
    t = re.sub(r'%\s*|\b%\b', ' százalék ', t)
    t = re.sub(r'\bTV\b', ' Tévé', t)
    t = re.sub(r'\bUSA\b', ' U S A', t)

    def num_to_hu(n: int) -> str:
        if n == 0:
            return "nulla"
        units = ["", "egy", "kettő", "három", "négy", "öt", "hat", "hét", "nyolc", "kilenc"]
        tens_exact = ["", "tíz", "húsz", "harminc", "negyven", "ötven", "hatvan", "hetven", "nyolcvan", "kilencven"]
        tens_prefix = ["", "tizen", "huszon", "harminc", "negyven", "ötven", "hatvan", "hetven", "nyolcvan", "kilencven"]

        if n < 10:
            return units[n]
        if n < 100:
            d1, d2 = divmod(n, 10)
            if d2 == 0:
                return tens_exact[d1]
            return tens_prefix[d1] + units[d2]
        if n < 1000:
            d1, d2 = divmod(n, 100)
            prefix = ("egy" if d1 == 1 else units[d1]) + "száz"
            return prefix if d2 == 0 else prefix + num_to_hu(d2)
        if n < 1000000:
            d1, d2 = divmod(n, 1000)
            prefix = num_to_hu(d1) + "ezer"
            return prefix if d2 == 0 else prefix + ("-" if d1 > 2 else "") + num_to_hu(d2)
        return str(n)

    months_hu = ["", "január", "február", "március", "április", "május", "június", "július", "augusztus", "szeptember", "október", "november", "december"]

    def date_repl(match):
        try:
            y, m, d = int(match.group(1)), int(match.group(2)), int(match.group(3))
            y_str = num_to_hu(y)
            m_str = months_hu[m] if 1 <= m <= 12 else str(m)
            d_str = "első" if d == 1 else ("második" if d == 2 else num_to_hu(d) + "adik")
            return f"{y_str} {m_str} {d_str}"
        except Exception:
            return match.group(0)

    t = re.sub(r'(\d{4})\.(\d{1,2})\.(\d{1,2})\.?', date_repl, t)

    def number_repl(match):
        try:
            val = int(match.group(0))
            if val <= 999999:
                return " " + num_to_hu(val) + " "
        except Exception:
            pass
        return match.group(0)

    t = re.sub(r'\b\d+\b', number_repl, t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


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

    vocab_path = "/workspace/vocab.txt"

    checkpoints = [
        "/workspace/model_927900.safetensors",
        "/workspace/model_927900.pt",
        "/workspace/model_309300.safetensors",
        "/workspace/model_309300.pt",
        "/workspace/model_122000-hun.pt",
        "/workspace/model_last_final.safetensors"
    ]

    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            try:
                print(f"Próbálkozás modell betöltésével: {ckpt}...", flush=True)
                tts_model = F5TTS(
                    ckpt_file=ckpt,
                    vocab_file=vocab_path if os.path.exists(vocab_path) else "",
                    device=device,
                    use_ema=True,
                )
                print(f"Sikeres modellbetöltés a helyi fájlból ({ckpt})!", flush=True)
                return tts_model
            except Exception as e:
                print(f"Figyelem: {ckpt} betöltési hiba: {e}", flush=True)

    try:
        print(f"Modellfájlok letöltése és betöltése a Hugging Face Hub-ról ({MODEL_NAME})...", flush=True)
        from huggingface_hub import hf_hub_download

        downloaded_ckpt = None
        downloaded_vocab = None

        try:
            downloaded_vocab = hf_hub_download(repo_id=MODEL_NAME, filename="vocab.txt")
            print(f"Sikeres vocab.txt letöltés/elérés: {downloaded_vocab}", flush=True)
        except Exception as ve:
            print(f"Figyelem: nem sikerült letölteni a vocab.txt-t a {MODEL_NAME} repóból: {ve}", flush=True)
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
                print(f"Letöltési kísérlet: {fname}...", flush=True)
                downloaded_ckpt = hf_hub_download(repo_id=MODEL_NAME, filename=fname)
                print(f"Sikeres {fname} letöltés: {downloaded_ckpt}", flush=True)
                break
            except Exception as de:
                print(f"Nem sikerült a(z) {fname} letöltése a {MODEL_NAME} repóból: {de}", flush=True)

        if not downloaded_ckpt:
            raise ValueError(f"Egyetlen ismert modellfájlt sem sikerült letölteni a {MODEL_NAME} repóból!")

        print(f"Próbálkozás modell betöltésével a letöltött fájlból ({downloaded_ckpt})...", flush=True)
        tts_model = F5TTS(
            ckpt_file=downloaded_ckpt,
            vocab_file=downloaded_vocab if downloaded_vocab else "",
            device=device,
            use_ema=True,
        )
        print(f"Modell ({MODEL_NAME}) sikeresen betöltve a letöltött fájlból!", flush=True)
        return tts_model

    except Exception as e:
        print(f"Modell betöltési hiba a HF Hub-ról: {e}", flush=True)
        raise e


def handler(event):
    print("Új feladat érkezett a dolgozóhoz!", flush=True)
    try:
        model = get_model()
    except Exception as e:
        err_msg = f"Modell inicializálási hiba: {e}\n{traceback.format_exc()}"
        print(err_msg, flush=True)
        return {"status": "error", "error": f"Modell inicializálási hiba: {e}"}

    try:
        input_data = event.get("input", {})
        text = input_data.get("text", "")
        ref_audio_base64 = input_data.get("ref_audio_base64") or input_data.get("ref_audio") or ""
        ref_audio_url = input_data.get("ref_audio_url", "")
        ref_text_in = input_data.get("ref_text", "")

        if not text or not str(text).strip():
            return {"status": "error", "error": "Hiányzó vagy üres 'text' paraméter!"}

        raw_text = str(text).strip()
        gen_text = normalize_hungarian_text(raw_text)
        print(f"Szövegnormalizálás elvégezve: '{raw_text[:40]}...' -> '{gen_text[:40]}...'", flush=True)

        ref_file = None

        if ref_audio_base64 and str(ref_audio_base64).strip():
            try:
                b64_str = str(ref_audio_base64).strip()
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                audio_bytes = base64.b64decode(b64_str)
                ref_file = "/tmp/user_ref_audio.wav"
                with open(ref_file, "wb") as f:
                    f.write(audio_bytes)
                print(f"Referencia hang dekódolva base64-ből ({len(audio_bytes)} bájt)", flush=True)
            except Exception as b64_err:
                print(f"Hiba a ref_audio_base64 dekódolásakor: {b64_err}", flush=True)

        if not ref_file and ref_audio_url and str(ref_audio_url).strip():
            ref_url = str(ref_audio_url).strip()
            if ref_url.startswith("http://") or ref_url.startswith("https://"):
                try:
                    print(f"Referencia hang letöltése URL-ről: {ref_url}", flush=True)
                    r = requests.get(ref_url, timeout=10)
                    if r.status_code == 200:
                        ref_file = "/tmp/user_ref_url.wav"
                        with open(ref_file, "wb") as f:
                            f.write(r.content)
                        print("Referencia hang sikeresen letöltve URL-ről.", flush=True)
                except Exception as url_err:
                    print(f"Hiba az URL letöltésekor: {url_err}", flush=True)

        if not ref_file or not os.path.exists(ref_file) or os.path.getsize(ref_file) < 100:
            print("Alapértelmezett referencia hang használata...", flush=True)
            ref_file = ensure_default_ref_audio()

        ref_text = DEFAULT_REF_TEXT
        if ref_text_in and str(ref_text_in).strip():
            ref_text = normalize_hungarian_text(str(ref_text_in).strip())

        print(f"F5TTS.infer hívás (sarpba/F5-TTS_V1_hun_v2): ref_file='{ref_file}', ref_text='{ref_text[:40]}...', gen_text='{gen_text[:40]}...'", flush=True)

        with torch.no_grad():
            res = model.infer(
                ref_file=ref_file,
                ref_text=ref_text,
                gen_text=gen_text,
                file_wave_name="/tmp/output.wav",
                seed=-1
            )

        output_path = "/tmp/output.wav"
        if not os.path.exists(output_path):
            if isinstance(res, tuple) and len(res) > 0 and isinstance(res[0], str) and os.path.exists(res[0]):
                output_path = res[0]
            elif isinstance(res, str) and os.path.exists(res):
                output_path = res

        if not os.path.exists(output_path):
            return {"status": "error", "error": "A generált hangfájl nem található a lemezen!"}

        with open(output_path, "rb") as f:
            audio_data = f.read()

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")
        print(f"Sikeres beszédgenerálás! Audio méret: {len(audio_data)} bájt", flush=True)

        return {
            "status": "success",
            "audio_base64": audio_b64,
            "format": "wav",
            "sample_rate": 24000
        }

    except Exception as e:
        err_str = f"Inferencia hiba történt: {e}\n{traceback.format_exc()}"
        print(err_str, flush=True)
        return {"status": "error", "error": f"Generálási hiba: {e}"}

print("Calling runpod.serverless.start...", flush=True)
runpod.serverless.start({"handler": handler})
