"""OWL-ViT, finding objects of candidate labels with `transformers`' zero-shot object detection pipeline, as `verify.rs` finds them in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def zero_shot_object_detection(snapshot):
    detect = pipeline(
        "zero-shot-object-detection", model=snapshot, dtype=torch.float32, device=DEVICE
    )
    return lambda inputs, parameters: [
        detect(item["image"], candidate_labels=item["candidate_labels"], **parameters)
        for item in inputs
    ]


verification = Verification(__file__)
verification.check("detection", zero_shot_object_detection)
verification.finish()
