"""Swin2SR, doubling images with `transformers`' image processor and model, rounded to bytes as its documentation shows, which no `transformers` pipeline serves, as `verify.rs` doubles them in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution  # noqa: E402

from support import DEVICE, Verification, data_url  # noqa: E402


def image_to_image(snapshot):
    processor = AutoImageProcessor.from_pretrained(snapshot)
    model = (
        Swin2SRForImageSuperResolution.from_pretrained(snapshot, dtype=torch.float32)
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
