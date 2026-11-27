"""LongT5, summarizing a text long enough to leave its local band and fill several global blocks, and one of about 15k tokens, near the 16k the checkpoint was trained on (a story around six copies of the Apache License), with `transformers`' tokenizer and `generate`, as `mod.rs` summarizes it in Singe.
`generate` reads the rest of its settings from the config, as Singe does: two beams, a length penalty, the n-gram ban, `repetition_penalty` 3.5, and `encoder_no_repeat_ngram_size` 4."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from support import DEVICE, Verification


def summarization(snapshot, dtype):
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(snapshot, dtype=dtype).to(DEVICE).eval()
    )

    def run(texts, parameters):
        outputs = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=parameters["max_new_tokens"],
                )
            outputs.append(
                {
                    "summary_text": tokenizer.decode(
                        generated[0], skip_special_tokens=True
                    )
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("summary", summarization)
verification.check("book", summarization)
verification.finish()
