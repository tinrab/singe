"""InstructPix2Pix, editing an image by an instruction with `diffusers`' `StableDiffusionInstructPix2PixPipeline`, as `mod.rs` edits it in Singe.
Both draw the noise from PyTorch's CPU generator seeded alike; the pipeline runs without its safety checker, which Singe's has none of."""

import numpy
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image

from support import DEVICE, Verification, data_url


def image_text_to_image(snapshot, dtype):
    edit = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        snapshot,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(DEVICE)
    edit.set_progress_bar_config(disable=True)

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            image = edit(
                item["prompt"],
                image=Image.open(item["image"]).convert("RGB"),
                generator=torch.Generator("cpu").manual_seed(item["seed"]),
                **parameters,
            ).images[0]
            outputs.append(data_url(numpy.asarray(image)))
        return outputs

    return run


verification = Verification(__file__)
verification.check("edit", image_text_to_image)
verification.finish()
