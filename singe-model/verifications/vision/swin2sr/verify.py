"""Swin2SR, doubling images with `transformers`' image processor and model, rounded to bytes as its documentation shows, which no `transformers` pipeline serves, as `mod.rs` doubles them in Singe."""

import numpy
import torch
from PIL import Image
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

from support import DEVICE, Verification, data_url


def image_to_image(snapshot, dtype):
    processor = AutoImageProcessor.from_pretrained(snapshot)
    model = (
        Swin2SRForImageSuperResolution.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )

    def run(paths, parameters):
        outputs = []
        for path in paths:
            inputs = processor(Image.open(path).convert("RGB"), return_tensors="pt").to(
                DEVICE
            )
            with torch.no_grad():
                reconstruction = model(**inputs).reconstruction
            pixels = reconstruction.squeeze(0).float().cpu().clamp_(0, 1).numpy()
            pixels = (numpy.moveaxis(pixels, 0, -1) * 255.0).round().astype(numpy.uint8)
            outputs.append(data_url(pixels))
        return outputs

    return run


verification = Verification(__file__)
verification.check("upscale", image_to_image)
verification.finish()
