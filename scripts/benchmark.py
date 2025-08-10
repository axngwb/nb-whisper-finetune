import argparse
import time
from pathlib import Path
import numpy as np
import torchaudio
from transformers import pipeline


def load_audio(path: str, sr: int = 16000):
    wav, s = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if s != sr:
        wav = torchaudio.transforms.Resample(s, sr)(wav)
    return {"array": wav.squeeze(0).numpy().astype(np.float32), "sampling_rate": sr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=str)
    ap.add_argument("--model_path", default="outputs/whisper-finetuned")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    asr = pipeline(
        task="automatic-speech-recognition",
        model=args.model_path,
        device=args.device,
        chunk_length_s=28,
        return_timestamps=False,
        generate_kwargs={"task": "transcribe", "language": "no", "num_beams": args.num_beams},
    )

    audio = load_audio(args.audio)
    for _ in range(args.warmup):
        _ = asr(audio)
    times = []
    for _ in range(args.runs):
        t0 = time.time()
        out = asr(audio)
        dt = time.time() - t0
        times.append(dt)
    rtf = np.mean(times) / (len(audio["array"]) / audio["sampling_rate"])
    print(f"Latency_avg_s={np.mean(times):.3f}, Latency_std_s={np.std(times):.3f}, RTF={rtf:.3f}")


if __name__ == "__main__":
    main()


