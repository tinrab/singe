"""SAM 3, finding what a text names in an image with `Sam3Processor`, `Sam3Model`, and `post_process_instance_segmentation`, which `transformers` has no pipeline for, as `mod.rs` finds it in Singe through image segmentation's `text` parameter.
The weights are `tiny-random/sam3`'s, random in `facebook/sam3`'s layout, since the real checkpoint is gated.
Its queries collapse to one score and one mask whose logits all sit just above zero, so the case keeps every query and cuts the masks at their median probability, which keeps the mask's shape in the comparison."""

import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from support import DEVICE, Verification, mask_url


def image_segmentation(snapshot, dtype):
    processor = Sam3Processor.from_pretrained(snapshot)
    model = Sam3Model.from_pretrained(snapshot, dtype=dtype).to(DEVICE).eval()

    def run(paths, parameters):
        outputs = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            inputs = processor(
                images=image, text=parameters["text"], return_tensors="pt"
            ).to(DEVICE)
            with torch.no_grad():
                output = model(**inputs)
            [found] = processor.post_process_instance_segmentation(
                output,
                threshold=parameters["threshold"],
                mask_threshold=parameters["mask_threshold"],
                target_sizes=[(image.height, image.width)],
            )
            outputs.append(
                [
                    {
                        "score": float(score),
                        "label": parameters["text"],
                        "mask": mask_url(found_mask.cpu()),
                    }
                    for score, found_mask in zip(found["scores"], found["masks"])
                ]
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("concept", image_segmentation)
verification.finish()
