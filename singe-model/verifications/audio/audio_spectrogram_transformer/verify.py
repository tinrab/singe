"""The Audio Spectrogram Transformer, classifying a clip with `transformers`' audio-classification pipeline, as `verify.rs` classifies it in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification, read_wav  # noqa: E402


def audio_classification(snapshot):
    classify = pipeline(
        "audio-classification", model=snapshot, dtype=torch.float32, device=DEVICE
    )

    def run(paths, parameters):
        outputs = []
        for path in paths:
            samples, rate = read_wav(path)
            outputs.append(
                classify({"raw": samples, "sampling_rate": rate}, **parameters)
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("speech", audio_classification)
verification.finish()
