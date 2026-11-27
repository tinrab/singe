"""SAM 2, generating masks of everything in an image with `transformers`' mask-generation pipeline, and tracking an object through a clip from points or a box with `Sam2VideoModel`, which `transformers` has no pipeline for, as `mod.rs` generates and tracks them in Singe."""

from pathlib import Path

import torch
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor, pipeline

from support import DEVICE, Verification, mask_url


def mask_generation(snapshot, dtype):
    generate = pipeline("mask-generation", model=snapshot, dtype=dtype, device=DEVICE)

    def run(paths, parameters):
        outputs = []
        for path in paths:
            found = generate(path, **parameters)
            outputs.append(
                {
                    "masks": [mask_url(mask) for mask in found["masks"]],
                    "scores": [float(score) for score in found["scores"]],
                }
            )
        return outputs

    return run


def mask_tracking(snapshot, dtype):
    # The memory attention reads tens of thousands of keys, so TF32 convolutions would drift over frames.
    torch.backends.cudnn.allow_tf32 = False
    processor = Sam2VideoProcessor.from_pretrained(snapshot)
    model = Sam2VideoModel.from_pretrained(snapshot, dtype=dtype).to(DEVICE).eval()

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            frames = [
                Image.open(path).convert("RGB")
                for path in sorted(Path(item["video"]).glob("*.png"))
            ]
            session = processor.init_video_session(
                video=frames, inference_device=DEVICE, dtype=dtype
            )
            points = item.get("points")
            box = item.get("box")
            processor.add_inputs_to_inference_session(
                inference_session=session,
                frame_idx=item["frame"],
                obj_ids=1,
                input_points=[[points]] if points else None,
                input_labels=[[item.get("labels") or [1] * len(points)]]
                if points
                else None,
                input_boxes=[[box]] if box else None,
            )
            tracked = {"frames": [], "masks": [], "scores": []}
            with torch.no_grad():
                model(inference_session=session, frame_idx=item["frame"])
                reverse = parameters.get("reverse", False)
                propagation = model.propagate_in_video_iterator(
                    session,
                    start_frame_idx=item["frame"] if reverse else None,
                    reverse=reverse,
                )
                for output in propagation:
                    [found] = processor.post_process_masks(
                        [output.pred_masks],
                        original_sizes=[[session.video_height, session.video_width]],
                        mask_threshold=parameters.get("mask_threshold", 0.0),
                    )
                    tracked["frames"].append(output.frame_idx)
                    tracked["masks"].append(mask_url(found[0, 0].cpu()))
                    tracked["scores"].append(
                        float(torch.sigmoid(output.object_score_logits.float()).item())
                    )
            outputs.append(tracked)
        return outputs

    return run


verification = Verification(__file__)
verification.check("everything", mask_generation)
verification.check("tracking", mask_tracking)
verification.check("tracking-outside", mask_tracking)
verification.check("tracking-reverse", mask_tracking)
verification.check("tracking-box", mask_tracking)
verification.finish()
