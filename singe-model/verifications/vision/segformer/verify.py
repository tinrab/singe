"""SegFormer, segmenting images by label with `transformers`' image-segmentation pipeline, as `mod.rs` segments them in Singe."""

import numpy
from transformers import pipeline

from support import DEVICE, Verification, data_url


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
verification.check("semantic", image_segmentation)
verification.finish()
