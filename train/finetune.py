import argparse
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import evaluate
from datasets import load_dataset
import torchaudio
import numpy as np
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers import __version__ as transformers_version


@dataclass
class Args:
    model_id: str
    dataset: str
    dataset_eval: Optional[str]
    text_column: str
    audio_column: str
    language: str
    train_split: str
    eval_split: Optional[str]
    output_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    lr: float
    gradient_accumulation_steps: int
    fp16: bool
    bf16: bool
    max_steps: int | None
    seed: int
    resume_from_checkpoint: Optional[str]
    aug_prob: float
    aug_speed_min: float
    aug_speed_max: float
    aug_noise_snr_min: float
    aug_noise_snr_max: float


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="NbAiLab/nb-whisper-large")
    p.add_argument("--dataset", required=True, help="hf dataset path or json/csv file for train or train split")
    p.add_argument("--dataset_eval", default=None, help="optional json/csv file for eval split")
    p.add_argument("--text_column", default="text")
    p.add_argument("--audio_column", default="audio")
    p.add_argument("--language", default="no")
    p.add_argument("--train_split", default="train")
    p.add_argument("--eval_split", default=None)
    p.add_argument("--output_dir", default="outputs/whisper-finetuned")
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--aug_prob", type=float, default=0.0)
    p.add_argument("--aug_speed_min", type=float, default=0.9)
    p.add_argument("--aug_speed_max", type=float, default=1.1)
    p.add_argument("--aug_noise_snr_min", type=float, default=15.0)
    p.add_argument("--aug_noise_snr_max", type=float, default=25.0)
    a = p.parse_args()
    return Args(**vars(a))


def _maybe_speed_perturb(waveform: torch.Tensor, sr: int, speed_min: float, speed_max: float) -> tuple[torch.Tensor, int]:
    speed = float(np.random.uniform(low=speed_min, high=speed_max))
    new_sr = max(1000, int(sr * speed))
    wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=new_sr)(waveform)
    wav = torchaudio.transforms.Resample(orig_freq=new_sr, new_freq=sr)(wav)
    return wav, sr


def _maybe_add_noise(waveform: torch.Tensor, snr_db: float) -> torch.Tensor:
    x = waveform
    p_signal = float(x.pow(2).mean().item() + 1e-8)
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))
    noise = torch.randn_like(x) * np.sqrt(p_noise)
    return x + noise


def prepare_dataset(ds, processor: WhisperProcessor, text_column: str, audio_column: str, language: str,
                    is_training: bool, aug_prob: float, speed_min: float, speed_max: float,
                    noise_snr_min: float, noise_snr_max: float):
    def map_batch(batch: Dict) -> Dict:
        path = batch[audio_column]
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if is_training and aug_prob > 0.0 and np.random.rand() < aug_prob:
            if np.random.rand() < 0.5:
                waveform, sr = _maybe_speed_perturb(waveform, sr, speed_min, speed_max)
            else:
                snr_db = float(np.random.uniform(low=noise_snr_min, high=noise_snr_max))
                waveform = _maybe_add_noise(waveform, snr_db)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
            sr = 16000
        audio_array = waveform.squeeze(0).numpy()
        input_features = processor.feature_extractor(audio_array, sampling_rate=sr).input_features[0]
        input_features = np.asarray(input_features, dtype=np.float32)
        labels = processor.tokenizer(batch[text_column]).input_ids
        return {"input_features": input_features, "labels": labels}

    return ds.map(map_batch, remove_columns=ds.column_names)


def build_model_and_processor(model_id: str):
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.generation_config.language = "no"
    model.generation_config.task = "transcribe"
    return model, processor


def main():
    args = parse_args()
    model, processor = build_model_and_processor(args.model_id)

    if args.dataset.endswith(".json") or args.dataset.endswith(".csv"):
        ds_train = load_dataset("json" if args.dataset.endswith(".json") else "csv", data_files=args.dataset)["train"]
        ds_eval = None
        if args.dataset_eval is not None and (args.dataset_eval.endswith(".json") or args.dataset_eval.endswith(".csv")):
            ds_eval = load_dataset("json" if args.dataset_eval.endswith(".json") else "csv", data_files=args.dataset_eval)["train"]
    else:
        ds_all = load_dataset(args.dataset)
        ds_train = ds_all[args.train_split]
        ds_eval = None
        if args.eval_split is not None and args.eval_split in ds_all:
            ds_eval = ds_all[args.eval_split]

    train_ds = prepare_dataset(
        ds_train,
        processor,
        args.text_column,
        args.audio_column,
        args.language,
        True,
        args.aug_prob,
        args.aug_speed_min,
        args.aug_speed_max,
        args.aug_noise_snr_min,
        args.aug_noise_snr_max,
    )

    eval_ds = None
    if ds_eval is not None:
        eval_ds = prepare_dataset(
            ds_eval,
            processor,
            args.text_column,
            args.audio_column,
            args.language,
            False,
            0.0,
            args.aug_speed_min,
            args.aug_speed_max,
            args.aug_noise_snr_min,
            args.aug_noise_snr_max,
        )

    class SimpleSpeechSeq2SeqCollator:
        def __call__(self, features: List[Dict]):
            max_frames = max(np.asarray(f["input_features"]).shape[-1] for f in features)
            batch_inputs = []
            for f in features:
                x = np.asarray(f["input_features"], dtype=np.float32)
                pad = max_frames - x.shape[-1]
                if pad > 0:
                    x = np.pad(x, ((0, 0), (0, pad)), mode="constant")
                batch_inputs.append(x)
            input_features = torch.tensor(np.stack(batch_inputs), dtype=torch.float32)

            max_len = max(len(f["labels"]) for f in features)
            labels_list = []
            for f in features:
                ids = f["labels"]
                pad = max_len - len(ids)
                if pad > 0:
                    ids = ids + [-100] * pad
                labels_list.append(ids)
            labels = torch.tensor(labels_list, dtype=torch.long)
            return {"input_features": input_features, "labels": labels}

    data_collator = SimpleSpeechSeq2SeqCollator()
    wer = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer.compute(predictions=pred_str, references=label_str)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        logging_dir="runs",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=(-1 if args.max_steps is None else args.max_steps),
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
        logging_steps=50,
        report_to=["tensorboard"],
        run_name="finetune",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )

    if args.max_steps == 0:
        pass
    else:
        if args.resume_from_checkpoint is not None:
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            trainer.train()
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    model.generation_config.save_pretrained(args.output_dir)
    meta = {"args": asdict(args), "transformers_version": transformers_version, "torch_version": torch.__version__}
    with open(f"{args.output_dir}/training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


