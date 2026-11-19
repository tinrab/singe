"""SAM, generating masks of everything in an image with `transformers`' mask-generation pipeline, as `mod.rs` generates them in Singe."""

import numpy
from transformers import pipeline

from support import DEVICE, Verification, data_url


def mask_generation(snapshot, dtype):
    generate = pipeline("mask-generation", model=snapshot, dtype=dtype, device=DEVICE)

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
