"""DeBERTa-v3, an inference model classifying zero-shot and text pairs with `transformers`' pipelines, as `mod.rs` classifies them in Singe."""

from transformers import pipeline

from support import DEVICE, Verification


def zero_shot_classification(snapshot, dtype):
    classify = pipeline(
        "zero-shot-classification", model=snapshot, dtype=dtype, device=DEVICE
    )
    return lambda inputs, parameters: [
        classify(item["sequences"], item["candidate_labels"], **parameters)
        for item in inputs
    ]


def text_classification(snapshot, dtype):
    classify = pipeline(
        "text-classification", model=snapshot, dtype=dtype, device=DEVICE
    )
    return lambda inputs, parameters: [classify(item, **parameters) for item in inputs]


verification = Verification(__file__)
verification.check("zero-shot", zero_shot_classification)
verification.check("pairs", text_classification)
verification.check("long-pair", text_classification)
verification.finish()
