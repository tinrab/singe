"""Mimi, encoding a clip to all its codebooks' codes and decoding it back with `transformers`' feature extractor and `MimiModel`, which no `transformers` pipeline serves, as `mod.rs` reconstructs it in Singe."""

import torch
from transformers import AutoFeatureExtractor, MimiModel

from support import DEVICE, Verification, read_audio


def audio_to_audio(snapshot, dtype):
    extractor = AutoFeatureExtractor.from_pretrained(snapshot)
    model = MimiModel.from_pretrained(snapshot, dtype=dtype).to(DEVICE).eval()

    def run(paths, parameters):
        outputs = []
        for path in paths:
            samples, rate = read_audio(path)
            inputs = extractor(samples, sampling_rate=rate, return_tensors="pt").to(
                DEVICE
            )
            with torch.no_grad():
                audio = model(
                    inputs["input_values"], padding_mask=inputs["padding_mask"]
                ).audio_values[0, 0]
            outputs.append(
                {
                    "audio": audio.float().cpu().tolist(),
                    "sampling_rate": extractor.sampling_rate,
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("reconstruction", audio_to_audio)
verification.finish()
