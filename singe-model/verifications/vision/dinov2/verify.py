"""DINOv2, pooling each image's features, as `mod.rs` extracts them in Singe, on images of two sizes, whose positions it resizes."""

import numpy
from PIL import Image
from transformers import pipeline

from support import DEVICE, Verification


def image_feature_extraction(snapshot, dtype):
    extract = pipeline(
        "image-feature-extraction", model=snapshot, dtype=dtype, device=DEVICE
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
