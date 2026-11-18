"""Wav2Vec2, transcribing a clip by CTC with `transformers`' automatic-speech-recognition pipeline, as `verify.rs` transcribes it in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification, read_wav  # noqa: E402


def automatic_speech_recognition(snapshot):
    recognize = pipeline(
        "automatic-speech-recognition",
        model=snapshot,
        dtype=torch.float32,
        device=DEVICE,
    )

    def run(paths, parameters):
        outputs = []
        for path in paths:
            samples, rate = read_wav(path)
            outputs.append(
                {
                    "text": recognize(
                        {"raw": samples, "sampling_rate": rate}, **parameters
                    )["text"]
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("transcription", automatic_speech_recognition)
verification.finish()
