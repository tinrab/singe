"""Depth Anything V2, estimating each pixel's depth with `transformers`' depth-estimation pipeline, as `mod.rs` estimates it in Singe."""

import numpy
from transformers import pipeline

from support import DEVICE, Verification, data_url


def depth_estimation(snapshot, dtype):
    estimate = pipeline("depth-estimation", model=snapshot, dtype=dtype, device=DEVICE)

    def run(paths, parameters):
        outputs = []
        for path in paths:
            depth = numpy.asarray(estimate(path, **parameters)["depth"].convert("RGB"))
            outputs.append({"depth": data_url(depth)})
        return outputs

    return run


verification = Verification(__file__)
verification.check("depth", depth_estimation)
verification.finish()
