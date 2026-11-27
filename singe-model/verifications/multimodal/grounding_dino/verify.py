"""Grounding DINO, finding objects of candidate labels with `transformers`' zero-shot object detection pipeline, as `mod.rs` finds them in Singe."""

from transformers import pipeline

from support import DEVICE, Verification


def zero_shot_object_detection(snapshot, dtype):
    detect = pipeline(
        "zero-shot-object-detection", model=snapshot, dtype=dtype, device=DEVICE
    )
    return lambda inputs, parameters: [
        detect(item["image"], candidate_labels=item["candidate_labels"], **parameters)
        for item in inputs
    ]


verification = Verification(__file__)
verification.check("detection", zero_shot_object_detection)
verification.finish()
