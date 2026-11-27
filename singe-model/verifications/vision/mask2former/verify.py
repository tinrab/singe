"""Mask2Former, segmenting images into panoptic segments with `transformers`' image-segmentation pipeline, as `mod.rs` segments them in Singe.
The pipeline is handed the loaded model, since building it from the path passes `code_revision`, which `Mask2FormerForUniversalSegmentation` does not take."""

import numpy
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    pipeline,
)

from support import DEVICE, Verification, data_url


def image_segmentation(snapshot, dtype):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(snapshot, dtype=dtype)
    segment = pipeline(
        "image-segmentation",
        model=model,
        image_processor=AutoImageProcessor.from_pretrained(snapshot),
        device=DEVICE,
    )

    def run(paths, parameters):
        return [
            [
                {
                    "score": region["score"],
                    "label": region["label"],
                    "mask": data_url(numpy.asarray(region["mask"].convert("RGB"))),
                }
                for region in segment(path, **parameters)
            ]
            for path in paths
        ]

    return run


verification = Verification(__file__)
verification.check("panoptic", image_segmentation)
verification.finish()
