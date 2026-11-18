"""SuperPoint, detecting and describing keypoints with `transformers`' image processor, model, and `post_process_keypoint_detection`, which `transformers` has no pipeline for, as `verify.rs` detects them in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoImageProcessor, SuperPointForKeypointDetection  # noqa: E402

from support import DEVICE, Verification  # noqa: E402

# The keypoints whose descriptors a case records, the first of each image's, to keep `cases.json` small.
DESCRIBED = 16


def keypoint_detection(snapshot):
    processor = AutoImageProcessor.from_pretrained(snapshot)
    model = (
        SuperPointForKeypointDetection.from_pretrained(snapshot, dtype=torch.float32)
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
