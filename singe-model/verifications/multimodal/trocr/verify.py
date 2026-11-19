"""TrOCR, reading a printed line with `transformers`' image processor, `VisionEncoderDecoderModel`, and `generate`, as the image-to-text pipeline did before `transformers` 5 removed it, and as `mod.rs` reads it in Singe."""

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

from support import DEVICE, Verification


def image_to_text(snapshot, dtype):
    processor = AutoImageProcessor.from_pretrained(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = (
        VisionEncoderDecoderModel.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )

    def run(paths, parameters):
        outputs = []
        for path in paths:
            pixels = processor(
                Image.open(path).convert("RGB"), return_tensors="pt"
            ).pixel_values.to(DEVICE)
            with torch.no_grad():
                generated = model.generate(
                    pixels,
                    max_new_tokens=parameters["max_new_tokens"],
                    do_sample=False,
                    num_beams=1,
                )
            outputs.append(
                {
                    "generated_text": tokenizer.decode(
                        generated[0], skip_special_tokens=True
                    )
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("printed", image_to_text)
verification.finish()
