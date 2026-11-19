"""PIA, animating an image after a prompt, and after an empty one, with `diffusers`' `PIAPipeline` over Stable Diffusion 1.5 and PIA's condition adapter, as `mod.rs` animates it in Singe.
Both draw the noise from PyTorch's CPU generator seeded alike, and both run in F16, since the UNet and its adapter do not fit the GPU in F32.
Guidance stays low, as AnimateDiff's does: in F16, a higher guidance makes the clip chaotic enough that `diffusers` against itself with another attention kernel differs by more than the comparison allows."""

import numpy
import torch
from diffusers import MotionAdapter, PIAPipeline
from PIL import Image

from support import DEVICE, Verification, data_url


def loaded(snapshot, dtype):
    adapter = MotionAdapter.from_pretrained(
        f"{snapshot}/motion_adapter", torch_dtype=dtype
    )
    animate = PIAPipeline.from_pretrained(
        snapshot, motion_adapter=adapter, torch_dtype=dtype
    ).to(DEVICE)
    animate.set_progress_bar_config(disable=True)
    return animate


def animation(animate, item, prompt, parameters):
    frames = animate(
        image=Image.open(item["image"]).convert("RGB"),
        prompt=prompt,
        generator=torch.Generator("cpu").manual_seed(item["seed"]),
        **parameters,
    ).frames[0]
    return [data_url(numpy.asarray(frame)) for frame in frames]


def image_text_to_video(snapshot, dtype):
    animate = loaded(snapshot, dtype)
    return lambda inputs, parameters: [
        animation(animate, item, item["prompt"], parameters) for item in inputs
    ]


def image_to_video(snapshot, dtype):
    animate = loaded(snapshot, dtype)
    return lambda inputs, parameters: [
        animation(animate, item, "", parameters) for item in inputs
    ]


verification = Verification(__file__)
verification.check("animation", image_text_to_video)
verification.check("image", image_to_video)
verification.finish()
