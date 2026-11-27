"""TimeSformer, classifying a clip's frames with `transformers`' image processor and model, sampled as its video-classification pipeline samples them, as `mod.rs` classifies them in Singe."""

from pathlib import Path

import numpy
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForVideoClassification

from support import DEVICE, Verification


def video_classification(snapshot, dtype):
    processor = AutoImageProcessor.from_pretrained(snapshot)
    model = (
        AutoModelForVideoClassification.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )

    def run(directories, parameters):
        outputs = []
        for directory in directories:
            frames = sorted(Path(directory).glob("*.png"))
            count = model.config.num_frames
            # The pipeline's sampling: `count` frames spread over the first `count`.
            indices = numpy.linspace(0, count - 1, num=count, dtype=numpy.int64)
            video = [Image.open(frames[index]).convert("RGB") for index in indices]
            inputs = processor(video, return_tensors="pt").to(DEVICE)
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


verification = Verification(__file__)
verification.check("classification", video_classification)
verification.finish()
