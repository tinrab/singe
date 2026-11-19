"""DAC, encoding a clip to its codes and decoding it back with `transformers`' feature extractor and `DacModel`, which no `transformers` pipeline serves, as `mod.rs` reconstructs it in Singe."""

import torch
from transformers import AutoFeatureExtractor, DacModel

from support import DEVICE, Verification, read_wav


def audio_to_audio(snapshot, dtype):
    extractor = AutoFeatureExtractor.from_pretrained(snapshot)
    model = DacModel.from_pretrained(snapshot, dtype=dtype).to(DEVICE).eval()

    def run(paths, parameters):
        outputs = []
        for path in paths:
            samples, rate = read_wav(path)
            inputs = extractor(samples, sampling_rate=rate, return_tensors="pt").to(
                DEVICE
            )
            with torch.no_grad():
                audio = model(inputs["input_values"]).audio_values[0]
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
