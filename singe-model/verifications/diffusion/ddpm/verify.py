"""DDPM, generating a CIFAR-10 image from a seed with `diffusers`' `DDPMPipeline`, as `verify.rs` generates it in Singe.
Both draw the noise from PyTorch's CPU generator seeded alike."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy  # noqa: E402
import torch  # noqa: E402
from diffusers import DDPMPipeline  # noqa: E402

from support import DEVICE, Verification, data_url  # noqa: E402


def unconditional_image_generation(snapshot):
    generate = DDPMPipeline.from_pretrained(snapshot, torch_dtype=torch.float32).to(
        DEVICE
    )
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
