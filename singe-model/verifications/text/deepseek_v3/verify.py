"""DeepSeek-V3, through a tiny random checkpoint of a dense layer and a mixture-of-experts layer (256 experts in 8 groups, multi-head latent attention, and the stored multi-token-prediction layer the import skips), generating greedily after two prompts with `transformers`' `generate`, as `verify.rs` generates in Singe.
Random weights test the architecture whole; DeepSeek-V3's trained weights need FP8 loading."""

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
            outputs.append(
                generation(model, generated[0, ids.shape[1] :].tolist())
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("prompt", text_generation)
verification.check("story", text_generation)
verification.finish()
