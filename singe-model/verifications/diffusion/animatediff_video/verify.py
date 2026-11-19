"""AnimateDiff, redrawing a clip after a prompt with `diffusers`' `AnimateDiffVideoToVideoPipeline` over Stable Diffusion 1.5 and its motion adapter, as `mod.rs` redraws it in Singe.
Both draw the noise from PyTorch's CPU generator seeded alike, and both run in F16, since the UNet and its adapter do not fit the GPU in F32."""

from pathlib import Path

import numpy
import torch
from diffusers import AnimateDiffVideoToVideoPipeline, MotionAdapter
from PIL import Image

from support import DEVICE, Verification, data_url


def video_to_video(snapshot, dtype):
    adapter = MotionAdapter.from_pretrained(
        f"{snapshot}/motion_adapter", torch_dtype=dtype
    )
    redraw = AnimateDiffVideoToVideoPipeline.from_pretrained(
        snapshot, motion_adapter=adapter, torch_dtype=dtype
    ).to(DEVICE)
    redraw.set_progress_bar_config(disable=True)

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            video = [
                Image.open(path).convert("RGB")
                for path in sorted(Path(item["video"]).glob("*.png"))
            ]
            frames = redraw(
                video=video,
                prompt=item["prompt"],
                generator=torch.Generator("cpu").manual_seed(item["seed"]),
                **parameters,
            ).frames[0]
            outputs.append([data_url(numpy.asarray(frame)) for frame in frames])
        return outputs

    return run


verification = Verification(__file__)
verification.check("redraw", video_to_video)
verification.finish()
