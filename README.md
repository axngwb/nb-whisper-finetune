# NB-Whisper Finetuning Starter Kit

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Windows](https://img.shields.io/badge/Platform-Windows%2010%2F11-blue.svg)](https://www.microsoft.com/windows)

> **Toolkit to adapt NbAiLab/nb-whisper-large to your own Norwegian data**

A comprehensive, Windows-friendly pipeline for fine-tuning Norwegian Whisper models with minimal setup and maximum reproducibility.

## Table of Contents

- [Features](#features)
- [Target Audience](#target-audience)
- [Project Comparison](#project-comparison)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

| Feature | Description |
|---------|-------------|
| **End-to-End Pipeline** | Complete workflow: dataset preparation → fine-tuning → evaluation → inference |
| **Model Analysis** | Comprehensive model inspection and step-wise token generation tracing |
| **Windows Native** | PowerShell commands ready out of the box |
| **TensorBoard Integration** | Real-time training monitoring with detailed logs |
| **Minimal Dependencies** | Clean, focused implementation without bloat |
| **GPU & CPU Support** | Flexible deployment options for different hardware |

**Base Model**: [`NbAiLab/nb-whisper-large`](https://huggingface.co/NbAiLab/nb-whisper-large) (Apache-2.0)

## Target Audience

- **Windows Developers** seeking quick NB-Whisper fine-tuning on custom data
- **ML Practitioners** needing minimal, reproducible pipelines without heavy infrastructure
- **Researchers** wanting practical PyTorch implementation over complex research setups

## Project Comparison

| Aspect | Official nb-whisper | This Starter Kit |
|--------|-------------------|------------------|
| **Focus** | Research-oriented Flax/TPU training | Practical PyTorch Windows starter |
| **Dependencies** | Complex research stack | Minimal, focused dependencies |
| **Platform** | Multi-platform research | Windows-optimized development |
| **Use Case** | Experiments & research | Production-ready fine-tuning |

**Official Repository**: [NbAiLab/nb-whisper](https://github.com/NbAiLab/nb-whisper)

## System Requirements

| Component | Requirement | Notes |
|-----------|-------------|-------|
| **OS** | Windows 10/11 | Tested on both versions |
| **Python** | 3.10+ | Tested with 3.11 |
| **Hardware** | CPU (smoke tests) / GPU (CUDA 12.4) | GPU recommended for production |
| **Memory** | 8GB+ RAM | 16GB+ recommended for GPU training |

## Quick Start

### Step 1: Environment Setup

<details>
<summary><strong>Initial Setup</strong></summary>

```powershell
# Create virtual environment
py -3 -m venv .venv

# Install base requirements
./.venv/Scripts/pip install -r requirements.txt
```

**Choose PyTorch variant:**

```powershell
# CPU Version (for testing)
./.venv/Scripts/pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio

# GPU Version (recommended for training)
./.venv/Scripts/pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio
```
</details>

### Step 2: Verify Installation

```powershell
# Test base model inference
./.venv/Scripts/python scripts/asr_infer.py data/audio/king.mp3 --model_path NbAiLab/nb-whisper-large --device cpu --num_beams 5
```

### Step 3: Dataset Preparation

```powershell
# Auto-generate dataset from audio files
./.venv/Scripts/python scripts/prepare_dataset.py data/audio \
  --out_csv data/train.csv \
  --out_json data/train.json \
  --model NbAiLab/nb-whisper-large \
  --device cpu \
  --num_beams 5 \
  --val_fraction 0.1 \
  --out_csv_val data/val.csv \
  --out_json_val data/val.json
```

> **Important**: Edit `data/train.csv` to ensure transcript accuracy. Format: `audio,text`. Prefer clips ≤ 30s.

### Step 4: Training

<details>
<summary><strong>Quick Demo Run (CPU)</strong></summary>

```powershell
./.venv/Scripts/python train/finetune.py \
  --model_id NbAiLab/nb-whisper-large \
  --dataset data/train.csv \
  --max_steps 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --seed 42
```
</details>

<details>
<summary><strong>Production Training (GPU)</strong></summary>

```powershell
./.venv/Scripts/python train/finetune.py \
  --model_id NbAiLab/nb-whisper-large \
  --dataset data/train.csv \
  --dataset_eval data/val.csv \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --fp16 \
  --seed 42
```
</details>

**Outputs**: Model saved to `outputs/whisper-finetuned/`, logs in `runs/`

### Step 5: Monitoring & Evaluation

<details>
<summary><strong>TensorBoard Setup</strong></summary>

```powershell
# Install and launch TensorBoard
./.venv/Scripts/python -m pip install tensorboard
./.venv/Scripts/tensorboard --logdir runs --host 127.0.0.1 --port 6006
```
Open: [http://127.0.0.1:6006](http://127.0.0.1:6006)
</details>

<details>
<summary><strong>Model Evaluation</strong></summary>

```powershell
# Compare base vs fine-tuned models
./.venv/Scripts/python scripts/asr_infer.py data/audio/king.mp3 --model_path NbAiLab/nb-whisper-large --device cpu --num_beams 5
./.venv/Scripts/python scripts/asr_infer_finetuned.py data/audio/king.mp3 --model_path outputs/whisper-finetuned --device cpu --num_beams 5

# Evaluate WER on validation set
./.venv/Scripts/python scripts/eval_wer.py data/val.csv --model_path outputs/whisper-finetuned --device cpu --num_beams 5
```
</details>

## Detailed Usage

### Model Analysis Tools

<details>
<summary><strong>Model Inspection</strong></summary>

```powershell
# Generate model structure summary
./.venv/Scripts/python scripts/inspect_model.py \
  --model_path NbAiLab/nb-whisper-large \
  --out analysis_model_summary.txt

# Trace token generation process
./.venv/Scripts/python scripts/trace_generate.py data/audio/king.mp3 \
  --model_path NbAiLab/nb-whisper-large \
  --max_new_tokens 64 \
  --num_beams 5 \
  --out analysis_trace.txt
```
</details>

### Current Limitations

| Limitation | Description | Workaround |
|------------|-------------|------------|
| **Training Method** | Full-model fine-tuning only (no LoRA/PEFT) | Use `--max_steps` for quick experiments |
| **Audio Processing** | No speaker diarization or VAD | Preprocess audio externally if needed |
| **Performance** | Auto-transcription slow on CPU | Use GPU or prepare transcripts manually |
| **Data Augmentation** | Basic augmentations only | Extend `finetune.py` for advanced techniques |

## Project Structure

```
nb-whisper-finetune/
├── scripts/                      # Core functionality
│   ├── asr_infer.py              # Base model inference
│   ├── asr_infer_finetuned.py    # Fine-tuned model inference  
│   ├── prepare_dataset.py        # Dataset preparation & auto-transcription
│   ├── inspect_model.py          # Model structure analysis
│   ├── trace_generate.py         # Token generation tracing
│   └── eval_wer.py               # WER evaluation
├── train/
│   └── finetune.py               # Full-model fine-tuning pipeline
├── data/                         # Your datasets (gitignored)
├── outputs/                      # Fine-tuned models (gitignored)  
├── runs/                         # TensorBoard logs (gitignored)
├── requirements.txt              # Python dependencies
└── README.md                     # This guide
```

## Troubleshooting

<details>
<summary><strong>Command Line Issues</strong></summary>

| Issue | Solution |
|-------|----------|
| Commands not working | Run in **PowerShell** (`PS ...`), not Python REPL (`>>>`) |
| Virtual environment not activating | Ensure you're using `./.venv/Scripts/` prefix |
| Permission denied | Run PowerShell as Administrator if needed |

</details>

<details>
<summary><strong>Audio & Data Issues</strong></summary>

| Issue | Solution |
|-------|----------|
| Training unstable with long audio | Split clips to ≤ 30 seconds |
| Poor transcription quality | Manually review and edit `data/train.csv` |
| Out of memory errors | Reduce batch size or use gradient accumulation |

</details>

<details>
<summary><strong>Performance Issues</strong></summary>

| Issue | Solution |
|-------|----------|
| CPU training too slow | Use `--max_steps 5` for testing, GPU for production |
| TorchCodec warnings | `pip uninstall -y torchcodec` (we use torchaudio) |
| GPU out of memory | Reduce batch size or enable `--fp16` |

</details>

<details>
<summary><strong>System Issues</strong></summary>

| Issue | Solution |
|-------|----------|
| Windows symlinks warning | Harmless; optionally enable Developer Mode |
| HuggingFace cache issues | Clear cache: `rm -rf ~/.cache/huggingface/` |
| CUDA version mismatch | Reinstall PyTorch with correct CUDA version |

</details>

## License

### Code License
This project is licensed under **Apache-2.0** - see [LICENSE](LICENSE) file for details.

### Model License  
**Base Model**: [`NbAiLab/nb-whisper-large`](https://huggingface.co/NbAiLab/nb-whisper-large) (Apache-2.0)

### Attribution
- **Maintainer**: [axngwb](https://github.com/axngwb)
- **Base Model**: Norwegian AI Lab (NbAiLab)