"""ViT, classifying images with `transformers`' image-classification pipeline, as `verify.rs` classifies them in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def image_classification(snapshot):
    classify = pipeline(
        "image-classification", model=snapshot, dtype=torch.float32, device=DEVICE
    )
    return lambda paths, parameters: [classify(path, **parameters) for path in paths]


verification = Verification(__file__)
verification.check("classification", image_classification)
verification.finish()
