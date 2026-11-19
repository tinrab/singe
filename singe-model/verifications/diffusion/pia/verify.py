"""PIA, animating an image after a prompt, and after an empty one, with `diffusers`' `PIAPipeline` over Stable Diffusion 1.5 and PIA's condition adapter, as `verify.rs` animates it in Singe.
Both draw the noise from PyTorch's CPU generator seeded alike, and both run in F16, since the UNet and its adapter do not fit the GPU in F32.
Guidance stays low, as AnimateDiff's does: in F16, a higher guidance makes the clip chaotic enough that `diffusers` against itself with another attention kernel differs by more than the comparison allows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy  # noqa: E402
import torch  # noqa: E402
from diffusers import MotionAdapter, PIAPipeline  # noqa: E402
from PIL import Image  # noqa: E402

from support import DEVICE, Verification, data_url  # noqa: E402


def loaded(snapshot):
    adapter = MotionAdapter.from_pretrained(
        f"{snapshot}/motion_adapter", torch_dtype=torch.float16
    )
    animate = PIAPipeline.from_pretrained(
        snapshot, motion_adapter=adapter, torch_dtype=torch.float16
    ).to(DEVICE)
    animate.set_progress_bar_config(disable=True)
    return animate


def animation(animate, item, parameters):
    frames = animate(
        image=Image.open(item["image"]).convert("RGB"),
        prompt=item.get("prompt", ""),
        generator=torch.Generator("cpu").manual_seed(item["seed"]),
        **parameters,
    ).frames[0]
    return [data_url(numpy.asarray(frame)) for frame in frames]


def image_text_to_video(snapshot):
    animate = loaded(snapshot)
    return lambda inputs, parameters: [
        animation(animate, item, parameters) for item in inputs
    ]


verification = Verification(__file__)
verification.check("animation", image_text_to_video)
verification.check("image", image_text_to_video)
verification.finish()
