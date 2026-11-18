"""CLIP, classifying an image among candidate labels with `transformers`' zero-shot image classification pipeline, as `verify.rs` classifies it in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def zero_shot_image_classification(snapshot):
    classify = pipeline(
        "zero-shot-image-classification",
        model=snapshot,
        dtype=torch.float32,
        device=DEVICE,
    )
    return lambda inputs, parameters: [
        classify(item["image"], candidate_labels=item["candidate_labels"], **parameters)
        for item in inputs
    ]


verification = Verification(__file__)
verification.check("zero-shot", zero_shot_image_classification)
verification.finish()
