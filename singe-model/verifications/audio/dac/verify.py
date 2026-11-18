"""DAC, encoding a clip to its codes and decoding it back with `transformers`' feature extractor and `DacModel`, which no `transformers` pipeline serves, as `verify.rs` reconstructs it in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import AutoFeatureExtractor, DacModel  # noqa: E402

from support import DEVICE, Verification, read_wav  # noqa: E402


def audio_to_audio(snapshot):
    extractor = AutoFeatureExtractor.from_pretrained(snapshot)
    model = DacModel.from_pretrained(snapshot, dtype=torch.float32).to(DEVICE).eval()

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
