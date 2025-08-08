## NB‑Whisper Finetuning Starter Kit (Windows‑friendly)

Professional toolkit to adapt NbAiLab/nb‑whisper‑large to your own Norwegian data. Includes dataset prep, full‑model finetuning, base/finetuned inference, model inspection, token trace, and TensorBoard monitoring. Designed to run locally on Windows (CPU or GPU).

Repository focus
- **Base model**: `NbAiLab/nb-whisper-large` (Apache‑2.0). Model card: `https://huggingface.co/NbAiLab/nb-whisper-large`
- **Target**: practical, minimal, reproducible finetuning on your data

### Key features
- **End‑to‑end pipeline**: dataset prep → finetune → evaluate → inference
- **Model analysis**: structure summary and step‑wise token trace
- **Windows PowerShell commands** out of the box
- **TensorBoard** logs in `runs/`

### Who this is for
- **Windows developers** who want to quickly finetune NB‑Whisper on their own data
- **Practitioners** who need a minimal, reproducible pipeline without heavy infra

### How this differs from the official nb‑whisper
- Official: research‑oriented Flax/TPU training and experiments
- This kit: practical **PyTorch Windows starter** with minimal dependencies and ready‑to‑run commands
- Official repo: [NbAiLab/nb‑whisper](https://github.com/NbAiLab/nb-whisper)

### Compatibility
- Windows 10/11, Python 3.10+ (tested 3.11)
- CPU works for smoke tests; **GPU (CUDA 12.4)** recommended for real runs

## Quick start

### 1) Environment
```
py -3 -m venv .venv
./.venv/Scripts/pip install -r requirements.txt
# Install PyTorch (choose one):
# CPU
./.venv/Scripts/pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
# or GPU (CUDA 12.4)
./.venv/Scripts/pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio
```

### 2) Base inference (sanity check)
```
./.venv/Scripts/python scripts/asr_infer.py audio/king.mp3 --model_path NbAiLab/nb-whisper-large --device cpu --num_beams 5
```

### 3) Inspect model & token trace (optional)
```
./.venv/Scripts/python scripts/inspect_model.py --model_path NbAiLab/nb-whisper-large --out analysis_model_summary.txt
./.venv/Scripts/python scripts/trace_generate.py audio/king.mp3 --model_path NbAiLab/nb-whisper-large --max_new_tokens 64 --num_beams 5 --out analysis_trace.txt
```

### 4) Prepare your dataset
Put audio under `data/audio/`, then auto‑create CSV/JSON (draft transcripts via base model):
```
./.venv/Scripts/python scripts/prepare_dataset.py data/audio --out_csv data/train.csv --out_json data/train.json --model NbAiLab/nb-whisper-large --device cpu --num_beams 5 --val_fraction 0.1 --out_csv_val data/val.csv --out_json_val data/val.json
```
Edit `data/train.csv` to ensure accurate transcripts. Expected columns: `audio,text`. Prefer clips ≤ 30s.

### 5) Finetune (full model)
- Short demo run (CPU okay):
```
./.venv/Scripts/python train/finetune.py --model_id NbAiLab/nb-whisper-large --dataset data/train.csv --max_steps 5 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 --seed 42
```
- Real run (GPU recommended):
```
./.venv/Scripts/python train/finetune.py --model_id NbAiLab/nb-whisper-large --dataset data/train.csv --dataset_eval data/val.csv --num_train_epochs 1 --per_device_train_batch_size 4 --gradient_accumulation_steps 4 --fp16 --seed 42
```
Outputs: `outputs/whisper-finetuned/` (model + processor), logs in `runs/`.

### 6) Monitor training
```
./.venv/Scripts/python -m pip install tensorboard
./.venv/Scripts/tensorboard --logdir runs --host 127.0.0.1 --port 6006
```
Open `http://127.0.0.1:6006`.

### 7) Inference with your finetuned model
```
./.venv/Scripts/python scripts/asr_infer.py audio/king.mp3 --model_path NbAiLab/nb-whisper-large --device cpu --num_beams 5
./.venv/Scripts/python scripts/asr_infer_finetuned.py audio/king.mp3 --model_path outputs/whisper-finetuned --device cpu --num_beams 5
```

### 8) Evaluate WER on a CSV
```
./.venv/Scripts/python scripts/eval_wer.py data/val.csv --model_path outputs/whisper-finetuned --device cpu --num_beams 5
```

## Limitations
- No LoRA/PEFT in this starter; it performs full‑model finetuning
- No diarization/speaker labels, VAD or advanced data augmentations out of the box
- Auto‑transcription during dataset preparation can be slow on CPU

## Project structure
```
scripts/
  asr_infer.py           # Base model inference (torchaudio input)
  asr_infer_finetuned.py # Inference using finetuned model dir (path in --model_path)
  prepare_dataset.py     # Scan audio folder -> CSV/JSON; optional auto‑transcripts
  inspect_model.py       # Config/params/modules summary → analysis_model_summary.txt
  trace_generate.py      # Step‑wise top‑K tokens with log‑probs → analysis_trace.txt
train/
  finetune.py            # Full‑model finetuning with TensorBoard logs, seed, resume, basic augs
data/                    # Your audio and CSV/JSON (kept out of VCS by .gitignore)
outputs/                 # Finetuned model (kept out of VCS)
runs/                    # TensorBoard logs (kept out of VCS)
```

## Troubleshooting
- **PowerShell vs Python REPL**: run commands in PowerShell (prompt `PS ...`), not inside `>>>` Python REPL.
- **Long audio**: split into ≤ 30s clips for more stable training.
- **CPU is slow**: use `--max_steps` for smoke tests; prefer GPU with `--fp16`.
- **TorchCodec warnings**: this kit uses torchaudio. If you installed `torchcodec`, you can uninstall it:
```
./.venv/Scripts/python -m pip uninstall -y torchcodec
```
- **Windows symlinks warning in HF cache**: harmless; optional to enable Developer Mode.

## License & attribution
- Code: **Apache‑2.0** (see `LICENSE`).
- Base model: **NbAiLab/nb‑whisper‑large** (Apache‑2.0). Model card: `https://huggingface.co/NbAiLab/nb-whisper-large`

## Author
- Maintainer: **axngwb** (`https://github.com/axngwb`)