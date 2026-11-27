"""SigLIP and SigLIP 2 NaFlex, classifying images among candidate labels with `transformers`' zero-shot image classification pipeline, which pads the texts to 64 tokens and scores each label by sigmoid, as `mod.rs` classifies them in Singe."""

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
verification.check("portrait", zero_shot_image_classification)
verification.check("naflex", zero_shot_image_classification)
verification.check("naflex-portrait", zero_shot_image_classification)
verification.finish()
