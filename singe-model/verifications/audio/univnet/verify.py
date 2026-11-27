"""UnivNet, resynthesizing a clip from its spectrogram with `transformers`' feature extractor and `UnivNetModel`, which no `transformers` pipeline serves, as `mod.rs` resynthesizes it in Singe.
The noise is Singe's, `normal_noise` in `head/audio.rs` ported here, so both sides read the same noise."""

import math

import numpy
import torch
from transformers import AutoFeatureExtractor, UnivNetModel

from support import DEVICE, Verification, read_audio

MASK = (1 << 64) - 1


def normal_noise(seed, count):
    """Standard normal values from `seed`, by SplitMix64 and the Box–Muller transform, as Singe draws them."""
    state = seed

    def uniform():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & MASK
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK
        value ^= value >> 31
        return ((value >> 11) + 0.5) / (1 << 53)

    values = []
    for _ in range(count):
        first, second = uniform(), uniform()
        values.append(
            math.sqrt(-2.0 * math.log(first)) * math.cos(2.0 * math.pi * second)
        )
    return numpy.asarray(values, dtype=numpy.float32)


def audio_to_audio(snapshot, dtype):
    extractor = AutoFeatureExtractor.from_pretrained(snapshot)
    model = UnivNetModel.from_pretrained(snapshot, dtype=dtype).to(DEVICE).eval()

    def run(paths, parameters):
        outputs = []
        for path in paths:
            samples, rate = read_audio(path)
            inputs = extractor(
                samples, sampling_rate=rate, return_noise=False, return_tensors="pt"
            )
            features = inputs["input_features"].to(DEVICE, dtype)
            frames = features.shape[1]
            channels = model.config.model_in_channels
            noise = torch.from_numpy(
                normal_noise(parameters.get("seed", 0), frames * channels)
            ).reshape(1, frames, channels)
            with torch.no_grad():
                waveform = model(features, noise.to(DEVICE, dtype)).waveforms[0]
            outputs.append(
                {
                    "audio": waveform.float().cpu().tolist(),
                    "sampling_rate": extractor.sampling_rate,
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("resynthesis", audio_to_audio)
verification.finish()
