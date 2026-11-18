"""SAM, generating masks of everything in an image with `transformers`' mask-generation pipeline, as `verify.rs` generates them in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy  # noqa: E402
import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification, data_url  # noqa: E402


def mask_generation(snapshot):
    generate = pipeline(
        "mask-generation", model=snapshot, dtype=torch.float32, device=DEVICE
    )

    def run(paths, parameters):
        outputs = []
        for path in paths:
            found = generate(path, **parameters)
            outputs.append(
                {
                    "masks": [
                        data_url(
                            numpy.repeat(
                                numpy.asarray(mask, dtype=numpy.uint8)[..., None] * 255,
                                3,
                                axis=2,
                            )
                        )
                        for mask in found["masks"]
                    ],
                    "scores": [float(score) for score in found["scores"]],
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("everything", mask_generation)
verification.finish()
