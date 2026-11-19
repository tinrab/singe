"""SD-Turbo, generating an image from a prompt and a seed in one step with `diffusers`' `StableDiffusionPipeline`, as `mod.rs` generates it in Singe.
Both draw the noise from PyTorch's CPU generator seeded alike.
Both run in F16: in F32 its weights alone take 5 GB, which leaves no headroom on an 8 GB GPU shared with other processes."""

import numpy
import torch
from diffusers import StableDiffusionPipeline

from support import DEVICE, Verification, data_url


def text_to_image(snapshot, dtype):
    generate = StableDiffusionPipeline.from_pretrained(snapshot, torch_dtype=dtype).to(
        DEVICE
    )
    generate.set_progress_bar_config(disable=True)

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            image = generate(
                item["prompt"],
                generator=torch.Generator("cpu").manual_seed(item["seed"]),
                **parameters,
            ).images[0]
            outputs.append(data_url(numpy.asarray(image)))
        return outputs

    return run


verification = Verification(__file__)
verification.check("generation", text_to_image)
verification.finish()
