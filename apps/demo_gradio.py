import gradio as gr
import time
from typing import Optional
import numpy as np
import torchaudio
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def load_audio_to_array(file_path: str, target_sr: int = 16000) -> dict:
    wav, sr = torchaudio.load(file_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return {"array": wav.squeeze(0).numpy().astype(np.float32), "sampling_rate": target_sr}


def build_asr(model_path: str, device: str, num_beams: Optional[int], adapter_path: Optional[str]):
    generate_kwargs = {"task": "transcribe", "language": "no"}
    if num_beams is not None:
        generate_kwargs["num_beams"] = int(num_beams)
    if adapter_path is None or adapter_path.strip() == "":
        return pipeline(
            task="automatic-speech-recognition",
            model=model_path,
            device=device,
            chunk_length_s=28,
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )
    if PeftModel is None:
        raise RuntimeError("peft is required to use a LoRA adapter")
    processor = WhisperProcessor.from_pretrained(model_path)
    base = WhisperForConditionalGeneration.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.generation_config.language = "no"
    model.generation_config.task = "transcribe"
    return pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        chunk_length_s=28,
        return_timestamps=False,
        generate_kwargs=generate_kwargs,
    )


def transcribe(audio_file, model_path, use_adapter, adapter_path, device, num_beams):
    if audio_file is None:
        return "", 0.0
    t0 = time.time()
    asr = build_asr(model_path, device, num_beams, adapter_path if use_adapter else None)
    audio = load_audio_to_array(audio_file)
    out = asr(audio)
    dt = time.time() - t0
    return out["text"], dt


with gr.Blocks(title="NB-Whisper Demo") as demo:
    gr.Markdown("NB-Whisper ASR Demo (Norwegian). Supports base or fine-tuned models and optional LoRA adapters.")
    with gr.Row():
        audio = gr.Audio(sources=["upload"], type="filepath", label="Audio")
    with gr.Row():
        model_path = gr.Textbox(value="NbAiLab/nb-whisper-large", label="Model Path")
        use_adapter = gr.Checkbox(value=False, label="Use LoRA adapter")
        adapter_path = gr.Textbox(value="", label="Adapter Path (optional)")
    with gr.Row():
        device = gr.Dropdown(choices=["cpu", "cuda:0"], value="cpu", label="Device")
        num_beams = gr.Slider(minimum=1, maximum=8, value=5, step=1, label="Num Beams")
    btn = gr.Button("Transcribe")
    out_text = gr.Textbox(label="Transcript")
    latency = gr.Number(label="Latency (s)")

    btn.click(transcribe, inputs=[audio, model_path, use_adapter, adapter_path, device, num_beams], outputs=[out_text, latency])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)


