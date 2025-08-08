import argparse
from pathlib import Path
from transformers import pipeline
import torchaudio
import numpy as np


def load_audio(path: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return {"array": wav.squeeze(0).numpy().astype(np.float32), "sampling_rate": target_sr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=str)
    ap.add_argument("--model_path", default="outputs/whisper-finetuned")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_beams", type=int, default=5)
    args = ap.parse_args()

    asr = pipeline(
        task="automatic-speech-recognition",
        model=args.model_path,
        chunk_length_s=28,
        device=args.device,
        return_timestamps=True,
        generate_kwargs={"task": "transcribe", "language": "no", "num_beams": args.num_beams},
    )

    audio = load_audio(args.audio)
    result = asr(audio)
    print(result["text"])


if __name__ == "__main__":
    main()


