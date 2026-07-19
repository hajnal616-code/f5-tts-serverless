import os
import torch
import runpod
from f5_tts.infer.utils_infer import infer_process

def handler(job):
    job_input = job['input']
    text = job_input.get('text')
    ref_audio = job_input.get('ref_audio')
    
    # Ellenőrizzük, hogy a megadott fájl létezik-e a konténerben
    if not os.path.exists(ref_audio):
        return {"error": f"A megadott referenciahang nem található: {ref_audio}"}

    # Az F5-TTS inferencia futtatása
    # A kimenetet egy ideiglenes fájlba mentjük
    output_filename = "output.wav"
    output_path = infer_process(ref_audio, text, output_filename)
    
    return {"output_path": output_path}

# A RunPod szerverless futtatása
runpod.serverless.start({"handler": handler})
