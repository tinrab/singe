"""TAESD, encoding an image to latents and decoding it back with `diffusers`' `AutoencoderTiny`, as `mod.rs` reconstructs it in Singe."""

import numpy
import torch
from diffusers import AutoencoderTiny
from PIL import Image

from support import DEVICE, Verification, data_url


def image_to_image(snapshot, dtype):
    model = (
        AutoencoderTiny.from_pretrained(snapshot, torch_dtype=dtype).to(DEVICE).eval()
    )

    def run(paths, parameters):
        outputs = []
        for path in paths:
            pixels = numpy.asarray(Image.open(path).convert("RGB"))
            images = (
                torch.from_numpy(pixels).permute(2, 0, 1)[None].to(DEVICE, dtype)
                / 127.5
                - 1.0
            )
            with torch.no_grad():
                decoded = model.decode(model.encode(images).latents).sample
            reconstructed = (
                ((decoded[0].permute(1, 2, 0) + 1.0) * 127.5)
                .clamp(0, 255)
                .round()
                .byte()
            )
            outputs.append(data_url(reconstructed.cpu().numpy()))
        return outputs

    return run


verification = Verification(__file__)
verification.check("reconstruction", image_to_image)
verification.finish()
