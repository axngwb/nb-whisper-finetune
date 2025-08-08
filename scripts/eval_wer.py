import argparse
import csv
from pathlib import Path
from typing import List
import numpy as np
from transformers import pipeline
import torchaudio
import evaluate


def load_audio(path: str, target_sr: int = 16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return {"array": wav.squeeze(0).numpy().astype(np.float32), "sampling_rate": target_sr}


def read_csv(path: Path) -> List[tuple[str, str]]:
    rows: List[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["audio"], row["text"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str)
    ap.add_argument("--model_path", default="outputs/whisper-finetuned")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_beams", type=int, default=5)
    args = ap.parse_args()

    data = read_csv(Path(args.csv))

    asr = pipeline(
        task="automatic-speech-recognition",
        model=args.model_path,
        device=args.device,
        chunk_length_s=28,
        return_timestamps=False,
        generate_kwargs={"task": "transcribe", "language": "no", "num_beams": args.num_beams},
    )

    preds: List[str] = []
    refs: List[str] = []
    for audio_path, text in data:
        audio = load_audio(audio_path)
        out = asr(audio)
        preds.append(out["text"])
        refs.append(text)

    wer = evaluate.load("wer")
    score = wer.compute(predictions=preds, references=refs)
    print(f"WER: {score:.4f}")


if __name__ == "__main__":
    main()


