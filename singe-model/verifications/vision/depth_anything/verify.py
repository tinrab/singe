"""Depth Anything V2, estimating each pixel's depth with `transformers`' depth-estimation pipeline, as `verify.rs` estimates it in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy  # noqa: E402
import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification, data_url  # noqa: E402


def depth_estimation(snapshot):
    estimate = pipeline(
        "depth-estimation", model=snapshot, dtype=torch.float32, device=DEVICE
    )

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
