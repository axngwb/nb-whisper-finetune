import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple
import random
import numpy as np
import torchaudio
import soundfile as sf
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


def segment_audio_energy(waveform, sr: int, max_len_s: float, silence_thresh: float,
                         min_speech_s: float, min_silence_s: float) -> List[tuple[int, int]]:
    x = waveform.squeeze(0).numpy()
    frame_hop = int(sr * 0.02)
    frame_len = int(sr * 0.02)
    if frame_len <= 0:
        frame_len = 1
    if frame_hop <= 0:
        frame_hop = 1
    rms = []
    for i in range(0, len(x) - frame_len + 1, frame_hop):
        frame = x[i:i+frame_len]
        rms.append(float(np.sqrt(np.mean(frame * frame) + 1e-10)))
    if len(rms) == 0:
        return [(0, len(x))]
    thr = max(1e-8, np.median(rms) * silence_thresh)
    active = [v > thr for v in rms]
    segments = []
    in_seg = False
    seg_start = 0
    silence_frames = int(min_silence_s / 0.02)
    min_len_frames = int(min_speech_s / 0.02)
    i = 0
    while i < len(active):
        if not in_seg and active[i]:
            in_seg = True
            seg_start = i
        if in_seg:
            j = i
            silent = 0
            while j < len(active) and (active[j] or silent < silence_frames):
                if active[j]:
                    silent = 0
                else:
                    silent += 1
                j += 1
            seg_end = j
            if seg_end - seg_start >= min_len_frames:
                s = seg_start * frame_hop
                e = min(len(x), seg_end * frame_hop)
                max_samples = int(max_len_s * sr)
                k = s
                while k < e:
                    kk = min(e, k + max_samples)
                    segments.append((k, kk))
                    k = kk
            in_seg = False
            i = j
            continue
        i += 1
    if not segments:
        segments = [(0, len(x))]
    return segments


def maybe_segment_files(files: List[Path], out_segments_dir: Path, max_len_s: float,
                        silence_thresh: float, min_speech_s: float, min_silence_s: float) -> List[Path]:
    out_segments_dir.mkdir(parents=True, exist_ok=True)
    new_files: List[Path] = []
    for p in files:
        wav, sr = torchaudio.load(str(p))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        segs = segment_audio_energy(wav, sr, max_len_s, silence_thresh, min_speech_s, min_silence_s)
        base = p.stem
        for idx, (s, e) in enumerate(segs):
            arr = wav.squeeze(0).numpy()[s:e]
            if len(arr) < int(min_speech_s * sr):
                continue
            out_path = out_segments_dir / f"{base}_seg{idx:04d}.wav"
            sf.write(str(out_path), arr, sr)
            new_files.append(out_path)
    return new_files


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
    ap.add_argument("--use_vad", action="store_true", help="Energy-based segmentation into shorter chunks")
    ap.add_argument("--segment_max_seconds", type=float, default=28.0)
    ap.add_argument("--silence_threshold", type=float, default=1.5, help="Multiplier over median RMS")
    ap.add_argument("--min_speech_sec", type=float, default=1.0)
    ap.add_argument("--min_silence_sec", type=float, default=0.2)
    args = ap.parse_args()

    audio_root = Path(args.audio_dir)
    files = list_audio_files(audio_root)
    if not files:
        print("No audio files found.")
        return

    if args.use_vad:
        default_segments_dir = Path("data/segments")
        files = maybe_segment_files(
            files,
            default_segments_dir,
            max_len_s=float(args.segment_max_seconds),
            silence_thresh=float(args.silence_threshold),
            min_speech_s=float(args.min_speech_sec),
            min_silence_s=float(args.min_silence_sec),
        )
        if not files:
            print("Segmentation produced no segments, aborting.")
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


