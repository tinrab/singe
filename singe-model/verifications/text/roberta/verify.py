"""RoBERTa, through the tasks its snapshots here serve, as `mod.rs` runs them in Singe: filling masks, among targets or not, and an inference model classifying zero-shot and text pairs."""

from transformers import pipeline

from support import DEVICE, Verification


def fill_mask(snapshot, dtype):
    fill = pipeline("fill-mask", model=snapshot, dtype=dtype, device=DEVICE)
    return lambda texts, parameters: [fill(text, **parameters) for text in texts]


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
verification.check("fill", fill_mask)
verification.check("fill-targets", fill_mask)
verification.check("zero-shot", zero_shot_classification)
verification.check("pairs", text_classification)
verification.finish()
