"""AnimateDiff, redrawing a clip after a prompt with `diffusers`' `AnimateDiffVideoToVideoPipeline` over Stable Diffusion 1.5 and its motion adapter, as `verify.rs` redraws it in Singe.
Both draw the noise from PyTorch's CPU generator seeded alike, and both run in F16, since the UNet and its adapter do not fit the GPU in F32."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy  # noqa: E402
import torch  # noqa: E402
from diffusers import AnimateDiffVideoToVideoPipeline, MotionAdapter  # noqa: E402
from PIL import Image  # noqa: E402

from support import DEVICE, Verification, data_url  # noqa: E402


def video_to_video(snapshot):
    adapter = MotionAdapter.from_pretrained(
        f"{snapshot}/motion_adapter", torch_dtype=torch.float16
    )
    redraw = AnimateDiffVideoToVideoPipeline.from_pretrained(
        snapshot, motion_adapter=adapter, torch_dtype=torch.float16
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
