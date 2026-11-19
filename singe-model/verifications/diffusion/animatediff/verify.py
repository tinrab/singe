"""AnimateDiff, generating a clip from a prompt and a seed with `diffusers`' `AnimateDiffPipeline` over Stable Diffusion 1.5 and its motion adapter, as `mod.rs` generates it in Singe.
Both draw the noise from PyTorch's CPU generator seeded alike, and both run in F16, since the UNet and its adapter do not fit the GPU in F32.
Guidance stays low: at 256² in F16, `diffusers` against itself with another attention kernel differs by 33 dB at guidance 3 and by 47.5 dB at 1.5, so a higher guidance makes the clip too chaotic to compare."""

import numpy
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter

from support import DEVICE, Verification, data_url


def text_to_video(snapshot, dtype):
    adapter = MotionAdapter.from_pretrained(
        f"{snapshot}/motion_adapter", torch_dtype=dtype
    )
    generate = AnimateDiffPipeline.from_pretrained(
        snapshot, motion_adapter=adapter, torch_dtype=dtype
    ).to(DEVICE)
    generate.set_progress_bar_config(disable=True)

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            frames = generate(
                item["prompt"],
                generator=torch.Generator("cpu").manual_seed(item["seed"]),
                **parameters,
            ).frames[0]
            outputs.append([data_url(numpy.asarray(frame)) for frame in frames])
        return outputs

    return run


verification = Verification(__file__)
verification.check("generation", text_to_video)
verification.finish()
