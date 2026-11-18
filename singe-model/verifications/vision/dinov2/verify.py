"""DINOv2, pooling each image's features, as `verify.rs` extracts them in Singe, on images of two sizes, whose positions it resizes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def image_feature_extraction(snapshot):
    extract = pipeline(
        "image-feature-extraction", model=snapshot, dtype=torch.float32, device=DEVICE
    )

    def run(paths, parameters):
        outputs = []
        for path in paths:
            features = numpy.asarray(
                extract(Image.open(path).convert("RGB"), **parameters)
            )
            outputs.append(features.reshape(-1, features.shape[-1]).tolist())
        return outputs

    return run


verification = Verification(__file__)
verification.check("pooled", image_feature_extraction)
verification.finish()
