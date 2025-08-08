import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple
import random
import torchaudio
from transformers import pipeline


def list_audio_files(root: Path) -> List[Path]:
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def transcribe_files(files: List[Path], model_id: str, device: str, num_beams: int | None) -> List[str]:
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        device=device,
        chunk_length_s=28,
        return_timestamps=False,
        generate_kwargs={"task": "transcribe", "language": "no", **({"num_beams": num_beams} if num_beams else {})},
    )
    texts: List[str] = []
    for p in files:
        waveform, sr = torchaudio.load(str(p))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resample(waveform)
            sr = 16000
        audio_input = {"array": waveform.squeeze(0).numpy(), "sampling_rate": sr}
        out = asr(audio_input)
        texts.append(out["text"])
    return texts


def save_csv(files: List[Path], texts: List[str], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio", "text"])
        for p, t in zip(files, texts):
            writer.writerow([str(p).replace("\\", "/"), t])


def save_json(files: List[Path], texts: List[str], out_json: Path):
    out_json.parent.mkdir(parents=True, exist_ok=True)
    data = [{"audio": str(p).replace("\\", "/"), "text": t} for p, t in zip(files, texts)]
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_indices(n: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    idxs = list(range(n))
    random.Random(seed).shuffle(idxs)
    val_size = int(n * val_fraction)
    val = idxs[:val_size]
    train = idxs[val_size:]
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir", type=str)
    ap.add_argument("--model", default="NbAiLab/nb-whisper-large")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_beams", type=int, default=None)
    ap.add_argument("--out_csv", default="data/train.csv")
    ap.add_argument("--out_json", default="data/train.json")
    ap.add_argument("--out_csv_val", default="data/val.csv")
    ap.add_argument("--out_json_val", default="data/val.json")
    ap.add_argument("--val_fraction", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_auto_transcribe", action="store_true")
    args = ap.parse_args()

    audio_root = Path(args.audio_dir)
    files = list_audio_files(audio_root)
    if not files:
        print("No audio files found.")
        return

    if args.no_auto_transcribe:
        texts = [""] * len(files)
    else:
        texts = transcribe_files(files, args.model, args.device, args.num_beams)

    if args.val_fraction > 0.0:
        tr_idx, va_idx = split_indices(len(files), args.val_fraction, args.seed)
        files_tr = [files[i] for i in tr_idx]
        files_va = [files[i] for i in va_idx]
        texts_tr = [texts[i] for i in tr_idx]
        texts_va = [texts[i] for i in va_idx]
        save_csv(files_tr, texts_tr, Path(args.out_csv))
        save_json(files_tr, texts_tr, Path(args.out_json))
        save_csv(files_va, texts_va, Path(args.out_csv_val))
        save_json(files_va, texts_va, Path(args.out_json_val))
        print(f"Saved train: {args.out_csv}, {args.out_json}; val: {args.out_csv_val}, {args.out_json_val}")
    else:
        save_csv(files, texts, Path(args.out_csv))
        save_json(files, texts, Path(args.out_json))
        print(f"Saved: {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()


