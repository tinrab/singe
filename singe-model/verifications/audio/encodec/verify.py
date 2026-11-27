"""EnCodec, encoding a clip to its default bandwidth's codes, and to the largest bandwidth's, and decoding it back with `transformers`' feature extractor and `EncodecModel`, which no `transformers` pipeline serves, as `mod.rs` reconstructs it in Singe."""

import torch
from transformers import AutoFeatureExtractor, EncodecModel

from support import DEVICE, Verification, read_audio


def audio_to_audio(snapshot, dtype):
    extractor = AutoFeatureExtractor.from_pretrained(snapshot)
    model = EncodecModel.from_pretrained(snapshot, dtype=dtype).to(DEVICE).eval()

    def run(paths, parameters):
        outputs = []
        for path in paths:
            samples, rate = read_audio(path)
            inputs = extractor(samples, sampling_rate=rate, return_tensors="pt").to(
                DEVICE
            )
            with torch.no_grad():
                audio = model(
                    inputs["input_values"],
                    padding_mask=inputs["padding_mask"],
                    bandwidth=parameters.get("bandwidth"),
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
verification.check("full-bandwidth", audio_to_audio)
verification.finish()
