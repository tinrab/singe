"""SD-Turbo, generating an image from a prompt and a seed in one step with `diffusers`' `StableDiffusionPipeline`, as `verify.rs` generates it in Singe.
Both draw the noise from PyTorch's CPU generator seeded alike.
Both run in F16: in F32 its weights alone take 5 GB, which leaves no headroom on an 8 GB GPU shared with other processes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy  # noqa: E402
import torch  # noqa: E402
from diffusers import StableDiffusionPipeline  # noqa: E402

from support import DEVICE, Verification, data_url  # noqa: E402


def text_to_image(snapshot):
    generate = StableDiffusionPipeline.from_pretrained(
        snapshot, torch_dtype=torch.float16
    ).to(DEVICE)
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
