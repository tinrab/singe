"""BigBird, filling masks with `transformers`' fill-mask pipeline, as `mod.rs` fills them in Singe.
`fill` stays under 704 tokens, where `BigBirdModel` attends in full, and `long`, from the Apache License's text, runs past them, 942 and 1311 tokens, where it attends block-sparsely.
`BigBirdModel` switches to full attention for good on a short input, so each case loads its own pipeline."""

from transformers import pipeline

from support import DEVICE, Verification


def fill_mask(snapshot, dtype):
    fill = pipeline("fill-mask", model=snapshot, dtype=dtype, device=DEVICE)
    return lambda texts, parameters: [fill(text, **parameters) for text in texts]


verification = Verification(__file__)
verification.check("fill", fill_mask)
verification.check("long", fill_mask)
verification.finish()
