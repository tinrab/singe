"""T5, summarizing and translating with `transformers`' tokenizer and `generate`, after the prefixes its config's `task_specific_params` names, as the summarization and translation pipelines did before `transformers` 5 removed them, and as `verify.rs` does in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def generator(snapshot, task, key):
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(snapshot, dtype=torch.float32)
        .to(DEVICE)
        .eval()
    )

    def run(texts, parameters):
        name = (
            task
            if task == "summarization"
            else f"translation_{parameters['src_lang']}_to_{parameters['tgt_lang']}"
        )
        prefix = model.config.task_specific_params[name]["prefix"]
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
                {key: tokenizer.decode(generated[0], skip_special_tokens=True)}
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check(
    "summary", lambda snapshot: generator(snapshot, "summarization", "summary_text")
)
verification.check(
    "german", lambda snapshot: generator(snapshot, "translation", "translation_text")
)
verification.finish()
