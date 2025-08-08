import argparse
from pathlib import Path
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def summarize_modules(model: torch.nn.Module, depth: int = 1) -> list[str]:
    lines: list[str] = []
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        lines.append(f"{name}: {module.__class__.__name__} | params={params}")
        if depth > 1:
            for sub_name, sub_module in module.named_children():
                sub_params = sum(p.numel() for p in sub_module.parameters())
                lines.append(f"  {name}.{sub_name}: {sub_module.__class__.__name__} | params={sub_params}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="NbAiLab/nb-whisper-large")
    parser.add_argument("--out", default="analysis_model_summary.txt")
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path)

    total, trainable = count_parameters(model)
    cfg = model.config
    gen = model.generation_config
    tok = processor.tokenizer

    summary_lines = []
    summary_lines.append("=== Model Config ===")
    summary_lines.append(str({
        "model_type": cfg.model_type,
        "vocab_size": cfg.vocab_size,
        "d_model": cfg.d_model,
        "encoder_layers": cfg.encoder_layers,
        "decoder_layers": cfg.decoder_layers,
        "encoder_attention_heads": cfg.encoder_attention_heads,
        "decoder_attention_heads": cfg.decoder_attention_heads,
        "max_source_positions": cfg.max_source_positions,
        "max_target_positions": cfg.max_target_positions,
        "num_mel_bins": getattr(cfg, "num_mel_bins", None),
    }))

    summary_lines.append("\n=== Generation Config ===")
    summary_lines.append(str({
        "max_length": gen.max_length,
        "num_beams": gen.num_beams,
        "length_penalty": gen.length_penalty,
        "repetition_penalty": gen.repetition_penalty,
        "early_stopping": gen.early_stopping,
    }))

    summary_lines.append("\n=== Tokenizer ===")
    summary_lines.append(str({
        "name_or_path": tok.name_or_path,
        "vocab_size": tok.vocab_size,
        "pad_token": tok.pad_token,
        "bos_token": tok.bos_token,
        "eos_token": tok.eos_token,
        "unk_token": tok.unk_token,
        "special_tokens": tok.all_special_tokens[:20],
    }))

    summary_lines.append("\n=== Parameters ===")
    summary_lines.append(f"total={total:,} trainable={trainable:,}")

    summary_lines.append("\n=== Top-level Modules ===")
    summary_lines.extend(summarize_modules(model, depth=2))

    Path(args.out).write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()


