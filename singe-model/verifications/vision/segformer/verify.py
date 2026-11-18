"""SegFormer, segmenting images by label with `transformers`' image-segmentation pipeline, as `verify.rs` segments them in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy  # noqa: E402
import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification, data_url  # noqa: E402


def image_segmentation(snapshot):
    segment = pipeline(
        "image-segmentation", model=snapshot, dtype=torch.float32, device=DEVICE
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
verification.check("semantic", image_segmentation)
verification.finish()
