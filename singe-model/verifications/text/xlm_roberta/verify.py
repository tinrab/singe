"""XLM-RoBERTa, filling masks in two languages with `transformers`' fill-mask pipeline, as `verify.rs` fills them in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def fill_mask(snapshot):
    fill = pipeline("fill-mask", model=snapshot, dtype=torch.float32, device=DEVICE)
    return lambda texts, parameters: [fill(text, **parameters) for text in texts]


verification = Verification(__file__)
verification.check("fill", fill_mask)
verification.finish()
