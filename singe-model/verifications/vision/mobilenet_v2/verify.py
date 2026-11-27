"""MobileNetV2, classifying images with `transformers`' image-classification pipeline and segmenting them with DeepLabV3's checkpoint through the image-segmentation pipeline, as `mod.rs` does in Singe."""

import numpy
from transformers import pipeline

from support import DEVICE, Verification, data_url


def image_classification(snapshot, dtype):
    classify = pipeline(
        "image-classification", model=snapshot, dtype=dtype, device=DEVICE
    )
    return lambda paths, parameters: [classify(path, **parameters) for path in paths]


def image_segmentation(snapshot, dtype):
    segment = pipeline("image-segmentation", model=snapshot, dtype=dtype, device=DEVICE)

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
verification.check("classification", image_classification)
verification.check("segmentation", image_segmentation)
verification.finish()
