"""YOLOS, detecting objects with `transformers`' object-detection pipeline, as `verify.rs` detects them in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def object_detection(snapshot):
    detect = pipeline(
        "object-detection", model=snapshot, dtype=torch.float32, device=DEVICE
    )
    return lambda paths, parameters: [detect(path, **parameters) for path in paths]


verification = Verification(__file__)
verification.check("detection", object_detection)
verification.finish()
