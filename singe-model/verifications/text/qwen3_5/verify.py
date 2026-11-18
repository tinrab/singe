"""Qwen3.5, generating greedily after a prompt and about an image with `transformers`' `generate`, as `verify.rs` generates in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline  # noqa: E402

from support import DEVICE, Verification, generation  # noqa: E402


def text_generation(snapshot):
    processor = AutoProcessor.from_pretrained(snapshot)
    model = (
        AutoModelForImageTextToText.from_pretrained(snapshot, dtype=torch.float32)
        .to(DEVICE)
        .eval()
    )

    def run(prompts, parameters):
        outputs = []
        for prompt in prompts:
            ids = processor.tokenizer(
                prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(DEVICE)
            with torch.no_grad():
                generated = model.generate(
                    input_ids=ids,
                    max_new_tokens=parameters["max_new_tokens"],
                    do_sample=False,
                )
            outputs.append(
                generation(model, generated[0, ids.shape[1] :].tolist())
            )
        return outputs

    return run


def image_text_to_text(snapshot):
    processor = AutoProcessor.from_pretrained(snapshot)
    model = (
        AutoModelForImageTextToText.from_pretrained(snapshot, dtype=torch.float32)
        .to(DEVICE)
        .eval()
    )

    def run(chats, parameters):
        outputs = []
        for messages in chats:
            batch = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **parameters.get("chat_template_kwargs", {}),
            ).to(DEVICE)
            with torch.no_grad():
                generated = model.generate(
                    **batch,
                    max_new_tokens=parameters["max_new_tokens"],
                    do_sample=False,
                )
            outputs.append(
                generation(model, generated[0, batch["input_ids"].shape[1] :].tolist())
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("prompt", text_generation)
verification.check("image", image_text_to_text)


def any_to_any(snapshot):
    generate = pipeline(
        "any-to-any", model=snapshot, dtype=torch.float32, device=DEVICE
    )

    def run(chats, parameters):
        return [
            {
                "generated_text": generate(
                    text=messages,
                    max_new_tokens=parameters["max_new_tokens"],
                    do_sample=False,
                    return_full_text=False,
                    **parameters.get("chat_template_kwargs", {}),
                )[0]["generated_text"]
            }
            for messages in chats
        ]

    return run


verification.check("any", any_to_any)
verification.finish()
