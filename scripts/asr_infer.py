import argparse
from pathlib import Path
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
import torchaudio
import numpy as np
try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def build_pipeline(model_id: str, device: int | str, num_beams: int | None,
                   adapter_path: str | None = None):
    generate_kwargs = {"task": "transcribe", "language": "no"}
    if num_beams is not None:
        generate_kwargs["num_beams"] = int(num_beams)
    if adapter_path is None:
        return pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            device=device,
            chunk_length_s=28,
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )
    if PeftModel is None:
        raise RuntimeError("peft is required to use --adapter_path")
    processor = WhisperProcessor.from_pretrained(model_id)
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.generation_config.language = "no"
    model.generation_config.task = "transcribe"
    return pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        chunk_length_s=28,
        return_timestamps=True,
        generate_kwargs=generate_kwargs,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str, help="Path to audio file (wav/mp3/flac)")
    parser.add_argument("--model_path", type=str, default="NbAiLab/nb-whisper-large")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda:0")
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--adapter_path", type=str, default=None, help="Optional LoRA adapter directory")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    audio_input = {"array": waveform.squeeze(0).numpy().astype(np.float32), "sampling_rate": sample_rate}
    asr = build_pipeline(args.model_path, args.device, args.num_beams, args.adapter_path)
    result = asr(audio_input)
    print(result["text"])


if __name__ == "__main__":
    main()


