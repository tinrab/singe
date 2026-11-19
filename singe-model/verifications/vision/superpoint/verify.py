"""SuperPoint, detecting and describing keypoints with `transformers`' image processor, model, and `post_process_keypoint_detection`, which `transformers` has no pipeline for, as `mod.rs` detects them in Singe."""

import torch
from PIL import Image
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

from support import DEVICE, Verification

# The keypoints whose descriptors a case records, the first of each image's, to keep `cases.json` small.
DESCRIBED = 16


def keypoint_detection(snapshot, dtype):
    processor = AutoImageProcessor.from_pretrained(snapshot)
    model = (
        SuperPointForKeypointDetection.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )

    def run(paths, parameters):
        outputs = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                output = model(**inputs)
            [detected] = processor.post_process_keypoint_detection(
                output, [(image.height, image.width)]
            )
            outputs.append(
                {
                    "keypoints": detected["keypoints"].tolist(),
                    "scores": detected["scores"].tolist(),
                    "descriptors": detected["descriptors"][:DESCRIBED].tolist(),
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("keypoints", keypoint_detection)
verification.finish()
