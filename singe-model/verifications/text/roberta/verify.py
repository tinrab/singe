"""RoBERTa, through the tasks its snapshots here serve, as `verify.rs` runs them in Singe: filling masks, among targets or not, and an inference model classifying zero-shot and text pairs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def fill_mask(snapshot):
    fill = pipeline("fill-mask", model=snapshot, dtype=torch.float32, device=DEVICE)
    return lambda texts, parameters: [fill(text, **parameters) for text in texts]


def zero_shot_classification(snapshot):
    classify = pipeline(
        "zero-shot-classification", model=snapshot, dtype=torch.float32, device=DEVICE
    )
    return lambda inputs, parameters: [
        classify(item["sequences"], item["candidate_labels"], **parameters)
        for item in inputs
    ]


def text_classification(snapshot):
    classify = pipeline(
        "text-classification", model=snapshot, dtype=torch.float32, device=DEVICE
    )
    return lambda inputs, parameters: [classify(item, **parameters) for item in inputs]


verification = Verification(__file__)
verification.check("fill", fill_mask)
verification.check("fill-targets", fill_mask)
verification.check("zero-shot", zero_shot_classification)
verification.check("pairs", text_classification)
verification.finish()
