"""EfficientNet, classifying images with `transformers`' image-classification pipeline, as `mod.rs` classifies them in Singe."""

from transformers import pipeline

from support import DEVICE, Verification


def image_classification(snapshot, dtype):
    classify = pipeline(
        "image-classification", model=snapshot, dtype=dtype, device=DEVICE
    )
    return lambda paths, parameters: [classify(path, **parameters) for path in paths]


verification = Verification(__file__)
verification.check("classification", image_classification)
verification.finish()
