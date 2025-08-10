import numpy as np
from scripts.prepare_dataset import segment_audio_energy
import torch


def test_segment_audio_energy_basic():
    sr = 16000
    t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
    speech = 0.5 * np.sin(2 * np.pi * 220 * t)
    noise = 0.01 * np.random.randn(len(t))
    x = speech + noise
    wav = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    segs = segment_audio_energy(wav, sr, max_len_s=1.0, silence_thresh=1.5, min_speech_s=0.2, min_silence_s=0.1)
    assert len(segs) >= 1
    for s, e in segs:
        assert e > s

