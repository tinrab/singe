"""TrOCR, reading a printed line with `transformers`' image processor, `VisionEncoderDecoderModel`, and `generate`, as the image-to-text pipeline did before `transformers` 5 removed it, and as `verify.rs` reads it in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def image_to_text(snapshot):
    processor = AutoImageProcessor.from_pretrained(snapshot)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = (
        VisionEncoderDecoderModel.from_pretrained(snapshot, dtype=torch.float32)
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
