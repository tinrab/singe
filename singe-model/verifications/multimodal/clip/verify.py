"""CLIP, classifying an image among candidate labels with `transformers`' zero-shot image classification pipeline, as `mod.rs` classifies it in Singe."""

from transformers import pipeline

from support import DEVICE, Verification


def zero_shot_image_classification(snapshot, dtype):
    classify = pipeline(
        "zero-shot-image-classification",
        model=snapshot,
        dtype=dtype,
        device=DEVICE,
    )
    return lambda inputs, parameters: [
        classify(item["image"], candidate_labels=item["candidate_labels"], **parameters)
        for item in inputs
    ]


verification = Verification(__file__)
verification.check("zero-shot", zero_shot_image_classification)
verification.finish()
