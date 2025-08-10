import argparse
from pathlib import Path
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="outputs/whisper-finetuned")
    ap.add_argument("--out_dir", default="exports/onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
    processor = WhisperProcessor.from_pretrained(args.model_path)
    model.eval()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 80, 3000)
    torch.onnx.export(
        model,
        (dummy,),
        f"{args.out_dir}/whisper_encoder_decoder.onnx",
        input_names=["input_features"],
        output_names=["logits"],
        dynamic_axes={"input_features": {0: "batch", 2: "time"}, "logits": {0: "batch", 1: "time"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    processor.save_pretrained(args.out_dir)
    print(f"Saved ONNX to {args.out_dir}")


if __name__ == "__main__":
    main()


