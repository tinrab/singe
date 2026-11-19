"""Gemma 2, through a tiny random checkpoint whose one layer slides its window by the default layout and caps its scores, generating greedily after two prompts with `transformers`' `generate`, as `verify.rs` generates in Singe.
Random weights test the architecture whole; Gemma 2's smallest trained snapshot (2.6B) does not fit 8 GB in F32."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from support import DEVICE, Verification, generation  # noqa: E402


def text_generation(snapshot):
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = (
        AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.float32)
        .to(DEVICE)
        .eval()
    )

    def run(prompts, parameters):
        outputs = []
        for prompt in prompts:
            ids = tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(DEVICE)
            with torch.no_grad():
                generated = model.generate(
                    ids, max_new_tokens=parameters["max_new_tokens"], do_sample=False
                )
            outputs.append(generation(model, generated[0, ids.shape[1] :].tolist()))
        return outputs

    return run


verification = Verification(__file__)
verification.check("prompt", text_generation)
verification.check("story", text_generation)
verification.finish()
