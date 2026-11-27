"""Switch Transformers, summarizing with `transformers`' tokenizer and `generate` after its config's summarization prefix, as `mod.rs` does in Singe.
The input is long enough that more than 64 of its tokens choose one expert in some layer, which the run checks, so the capacity drops tokens."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from support import DEVICE, Verification


def summarization(snapshot, dtype):
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(snapshot, dtype=dtype).to(DEVICE).eval()
    )
    dropped = []

    def count(module, inputs, outputs):
        expert_index = outputs[0]
        if expert_index.shape[1] > 1:
            dropped.append(expert_index.shape[1] - int(expert_index.sum()))

    for name, module in model.named_modules():
        if name.startswith("encoder.") and name.endswith("router"):
            module.register_forward_hook(count)

    def run(texts, parameters):
        prefix = model.config.task_specific_params["summarization"]["prefix"]
        outputs = []
        for text in texts:
            inputs = tokenizer(prefix + text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=parameters["max_new_tokens"],
                    do_sample=False,
                    num_beams=1,
                )
            outputs.append(
                {
                    "summary_text": tokenizer.decode(
                        generated[0], skip_special_tokens=True
                    )
                }
            )
        assert sum(dropped) > 0, "no encoder layer dropped a token at capacity"
        return outputs

    return run


verification = Verification(__file__)
verification.check("summary", summarization)
verification.finish()
