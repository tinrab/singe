"""V-JEPA 2, classifying a clip's frames with `transformers`' video processor and model, every frame read as Singe's sampling of 16 frames reads them, and pooling a base checkpoint's encoder states over a clip, the mean of `get_vision_features`, which `transformers` has no pipeline for, as `mod.rs` classifies and extracts them in Singe."""

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoModelForVideoClassification, AutoVideoProcessor

from support import DEVICE, Verification


def video_classification(snapshot, dtype):
    processor = AutoVideoProcessor.from_pretrained(snapshot)
    model = (
        AutoModelForVideoClassification.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )

    def run(directories, parameters):
        outputs = []
        for directory in directories:
            frames = sorted(Path(directory).glob("*.png"))[
                : model.config.frames_per_clip
            ]
            video = [Image.open(frame).convert("RGB") for frame in frames]
            inputs = processor(video, do_sample_frames=False, return_tensors="pt").to(
                DEVICE
            )
            with torch.no_grad():
                probabilities = model(**inputs).logits[0].softmax(-1)
            scores, ids = probabilities.topk(parameters.get("top_k", 5))
            outputs.append(
                [
                    {"score": score, "label": model.config.id2label[i]}
                    for score, i in zip(scores.tolist(), ids.tolist())
                ]
            )
        return outputs

    return run


def video_feature_extraction(snapshot, dtype):
    processor = AutoVideoProcessor.from_pretrained(snapshot)
    model = AutoModel.from_pretrained(snapshot, dtype=dtype).to(DEVICE).eval()

    def run(directories, parameters):
        outputs = []
        for directory in directories:
            video = [
                Image.open(frame).convert("RGB")
                for frame in sorted(Path(directory).glob("*.png"))
            ]
            inputs = processor(video, do_sample_frames=False, return_tensors="pt").to(
                DEVICE
            )
            with torch.no_grad():
                states = model.get_vision_features(inputs["pixel_values_videos"])[0]
            if parameters.get("pool"):
                states = states.mean(0, keepdim=True)
            outputs.append(states.float().cpu().tolist())
        return outputs

    return run


verification = Verification(__file__)
verification.check("classification", video_classification)
verification.check("features", video_feature_extraction)
verification.finish()
