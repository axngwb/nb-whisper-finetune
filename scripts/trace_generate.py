import argparse
from pathlib import Path
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio


def load_audio(path: str, target_sr: int = 16000) -> dict:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    return {"array": waveform.squeeze(0).numpy().astype(np.float32), "sampling_rate": sr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=str)
    parser.add_argument("--model_path", default="NbAiLab/nb-whisper-large")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--out", default="analysis_trace.txt")
    args = parser.parse_args()

    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
    model.generation_config.language = "no"
    model.generation_config.task = "transcribe"

    audio = load_audio(args.audio)
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="no", task="transcribe")

    output = model.generate(
        inputs=inputs.input_features,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        return_dict_in_generate=True,
        output_scores=True,
        output_attentions=False,
        output_hidden_states=False,
        forced_decoder_ids=forced_decoder_ids,
    )

    sequences = output.sequences[0].tolist()
    scores = output.scores
    tokens = processor.batch_decode([sequences], skip_special_tokens=False)[0]

    lines = []
    lines.append("=== Generated tokens (raw ids) ===")
    lines.append(str(sequences))
    lines.append("\n=== Decoded (with specials) ===")
    lines.append(tokens)

    if scores is not None and len(scores) > 0:
        lines.append("\n=== Step-wise top-5 candidates ===")
        for step, step_scores in enumerate(scores):
            probs = torch.log_softmax(step_scores, dim=-1)
            topk = torch.topk(probs, k=5)
            items = []
            vals = topk.values
            idxs = topk.indices
            if vals.dim() == 2:
                vals = vals[0]
                idxs = idxs[0]
            for logp, tid in zip(vals.tolist(), idxs.tolist()):
                tid_int = int(tid)
                tok = processor.tokenizer.convert_ids_to_tokens(tid_int)
                items.append(f"{tid_int}:{tok}:{logp:.3f}")
            lines.append(f"t={step}: " + ", ".join(items))

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()


