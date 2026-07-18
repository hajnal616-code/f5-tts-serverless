import os
import torch
import runpod
from f5_tts.infer.utils_infer import infer_process

def handler(job):
    job_input = job['input']
    text = job_input.get('text')
    ref_audio = job_input.get('ref_audio') # Itt a fájl elérési útja vagy bázis64 kód

    # Itt hívjuk meg az F5-TTS inferencia funkcióját
    # A konkrét paramétereket a telepített verzió alapján kell majd pontosítani
    output_path = infer_process(ref_audio, text, "test_output.wav")

    return {"output_path": output_path}

runpod.serverless.start({"handler": handler})