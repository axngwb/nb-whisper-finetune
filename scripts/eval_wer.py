import argparse
import csv
from pathlib import Path
from typing import List
import numpy as np
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
import torchaudio
import evaluate
try:
    from peft import PeftModel
except Exception:
    PeftModel = None


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
    ap.add_argument("--adapter_path", type=str, default=None)
    args = ap.parse_args()

    data = read_csv(Path(args.csv))

    if args.adapter_path is None:
        asr = pipeline(
            task="automatic-speech-recognition",
            model=args.model_path,
            device=args.device,
            chunk_length_s=28,
            return_timestamps=False,
            generate_kwargs={"task": "transcribe", "language": "no", "num_beams": args.num_beams},
        )
    else:
        if PeftModel is None:
            raise RuntimeError("peft is required to use --adapter_path")
        processor = WhisperProcessor.from_pretrained(args.model_path)
        base = WhisperForConditionalGeneration.from_pretrained(args.model_path)
        model = PeftModel.from_pretrained(base, args.adapter_path)
        model.generation_config.language = "no"
        model.generation_config.task = "transcribe"
        asr = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
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


