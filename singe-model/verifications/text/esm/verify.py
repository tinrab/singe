"""ESM-2, filling masked residues of protein sequences with `transformers`' fill-mask pipeline, as `mod.rs` fills them in Singe."""

from transformers import pipeline

from support import DEVICE, Verification


def fill_mask(snapshot, dtype):
    fill = pipeline("fill-mask", model=snapshot, dtype=dtype, device=DEVICE)
    return lambda texts, parameters: [fill(text, **parameters) for text in texts]


verification = Verification(__file__)
verification.check("fill", fill_mask)
verification.check("fill-two", fill_mask)
verification.finish()
