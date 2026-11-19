"""DDPM, generating a CIFAR-10 image from a seed with `diffusers`' `DDPMPipeline`, as `mod.rs` generates it in Singe.
Both draw the noise from PyTorch's CPU generator seeded alike."""

import numpy
import torch
from diffusers import DDPMPipeline

from support import DEVICE, Verification, data_url


def unconditional_image_generation(snapshot, dtype):
    generate = DDPMPipeline.from_pretrained(snapshot, torch_dtype=dtype).to(DEVICE)
    generate.set_progress_bar_config(disable=True)

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            image = generate(
                generator=torch.Generator("cpu").manual_seed(item["seed"]),
                num_inference_steps=parameters["num_inference_steps"],
            ).images[0]
            outputs.append(data_url(numpy.asarray(image)))
        return outputs

    return run


verification = Verification(__file__)
verification.check("generation", unconditional_image_generation)
verification.finish()
