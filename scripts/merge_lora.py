import argparse
from transformers import WhisperForConditionalGeneration
try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="NbAiLab/nb-whisper-large")
    ap.add_argument("--adapter_path", required=True)
    ap.add_argument("--out_dir", default="outputs/whisper-merged")
    args = ap.parse_args()
    if PeftModel is None:
        raise RuntimeError("peft must be installed to merge LoRA")
    base = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    peft = PeftModel.from_pretrained(base, args.adapter_path)
    merged = peft.merge_and_unload()
    merged.save_pretrained(args.out_dir)
    print(f"Saved merged model to {args.out_dir}")


if __name__ == "__main__":
    main()


