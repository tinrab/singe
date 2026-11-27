"""Florence-2, answering task prompts about images with `transformers`' processor and `Florence2ForConditionalGeneration.generate`, as its model card runs it, as `mod.rs` answers them in Singe.
The answer is decoded with its special tokens, which hold the location tokens of boxes, and then without its start, end, and padding tokens, as `post_process_generation` cleans it before parsing; a task that locates things also records the answer `post_process_generation` parses."""

import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

from support import DEVICE, Verification


def image_to_text(snapshot, dtype):
    processor = AutoProcessor.from_pretrained(snapshot)
    model = (
        Florence2ForConditionalGeneration.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )

    def run(paths, parameters):
        outputs = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            inputs = processor(
                text=parameters["prompt"], images=image, return_tensors="pt"
            ).to(DEVICE, dtype)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=parameters["max_new_tokens"],
                    do_sample=False,
                )
            text = processor.batch_decode(generated, skip_special_tokens=False)[0]
            answer = (
                text.replace("<s>", "").replace("</s>", "").replace("<pad>", "").strip()
            )
            output = {"generated_text": answer}
            task = parameters["prompt"][: parameters["prompt"].index(">") + 1]
            if (
                processor.tasks_answer_post_processing_type.get(task, "pure_text")
                != "pure_text"
            ):
                output["parsed"] = processor.post_process_generation(
                    text, task=task, image_size=(image.width, image.height)
                )[task]
            outputs.append(output)
        return outputs

    return run


verification = Verification(__file__)
for name in [
    "caption",
    "ocr",
    "objects",
    "regions",
    "grounding",
    "segmentation",
    "proposals",
]:
    verification.check(name, image_to_text)
verification.finish()
