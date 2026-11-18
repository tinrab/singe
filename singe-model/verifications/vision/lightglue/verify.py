"""LightGlue, matching two images' keypoints with `transformers`' image processor, model, and `post_process_keypoint_matching`, sorted best first as its keypoint-matching pipeline sorts them, as `verify.rs` matches them in Singe.
The pipeline itself passes each image's `(width, height)` where the post-processing reads `(height, width)`, which scales keypoints of an image that is not square by the wrong sides, so this calls the steps it runs with each image's `(height, width)`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoImageProcessor, LightGlueForKeypointMatching  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def keypoint_matching(snapshot):
    processor = AutoImageProcessor.from_pretrained(snapshot)
    model = (
        LightGlueForKeypointMatching.from_pretrained(snapshot, dtype=torch.float32)
        .to(DEVICE)
        .eval()
    )

    def run(pairs, parameters):
        outputs = []
        for paths in pairs:
            images = [Image.open(path).convert("RGB") for path in paths]
            inputs = processor(images, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                output = model(**inputs)
            sizes = [[(image.height, image.width) for image in images]]
            [found] = processor.post_process_keypoint_matching(
                output, sizes, threshold=parameters.get("threshold", 0.0)
            )
            matches = [
                {
                    "keypoint_image_0": {"x": first[0].item(), "y": first[1].item()},
                    "keypoint_image_1": {"x": second[0].item(), "y": second[1].item()},
                    "score": score.item(),
                }
                for first, second, score in zip(
                    found["keypoints0"], found["keypoints1"], found["matching_scores"]
                )
            ]
            outputs.append(
                sorted(matches, key=lambda match: match["score"], reverse=True)
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("matches", keypoint_matching)
verification.finish()
