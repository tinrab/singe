"""AnimateDiff, generating a clip from a prompt and a seed with `diffusers`' `AnimateDiffPipeline` over Stable Diffusion 1.5 and its motion adapter, as `verify.rs` generates it in Singe.
Both draw the noise from PyTorch's CPU generator seeded alike, and both run in F16, since the UNet and its adapter do not fit the GPU in F32.
Guidance stays low: at 256² in F16, `diffusers` against itself with another attention kernel differs by 33 dB at guidance 3 and by 47.5 dB at 1.5, so a higher guidance makes the clip too chaotic to compare."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy  # noqa: E402
import torch  # noqa: E402
from diffusers import AnimateDiffPipeline, MotionAdapter  # noqa: E402

from support import DEVICE, Verification, data_url  # noqa: E402


def text_to_video(snapshot):
    adapter = MotionAdapter.from_pretrained(
        f"{snapshot}/motion_adapter", torch_dtype=torch.float16
    )
    generate = AnimateDiffPipeline.from_pretrained(
        snapshot, motion_adapter=adapter, torch_dtype=torch.float16
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
