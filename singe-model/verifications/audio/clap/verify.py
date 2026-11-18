"""CLAP, classifying a clip among candidate labels with `transformers`' zero-shot audio classification pipeline, as `verify.rs` classifies it in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification, read_wav  # noqa: E402


def zero_shot_audio_classification(snapshot):
    classify = pipeline(
        "zero-shot-audio-classification",
        model=snapshot,
        dtype=torch.float32,
        device=DEVICE,
    )

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            samples, _ = read_wav(item["audio"])
            outputs.append(
                classify(
                    samples, candidate_labels=item["candidate_labels"], **parameters
                )
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("sounds", zero_shot_audio_classification)
verification.finish()
