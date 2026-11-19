"""SmolVLM2, answering a question about an image with `transformers`' image-text-to-text pipeline, and describing an image and a clip with its processor and `generate`, as `mod.rs` does all three in Singe.
`transformers` has no pipeline for visual question answering or video-text-to-text: it answers visual questions through image-text-to-text, and a clip's frames are passed with their frame rate, which the processor samples them by."""

from pathlib import Path

import numpy
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline

from support import DEVICE, Verification, generation


def visual_question_answering(snapshot, dtype):
    generate = pipeline(
        "image-text-to-text", model=snapshot, dtype=dtype, device=DEVICE
    )

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text", "text": item["question"]},
                    ],
                }
            ]
            answer = generate(
                text=messages,
                max_new_tokens=parameters["max_new_tokens"],
                do_sample=False,
                return_full_text=False,
            )[0]["generated_text"]
            outputs.append({"answer": answer})
        return outputs

    return run


def image_text_to_text(snapshot, dtype):
    processor = AutoProcessor.from_pretrained(snapshot)
    model = (
        AutoModelForImageTextToText.from_pretrained(snapshot, dtype=dtype)
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


def video_text_to_text(snapshot, dtype):
    processor = AutoProcessor.from_pretrained(snapshot)
    model = (
        AutoModelForImageTextToText.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )

    def run(chats, parameters):
        outputs = []
        for messages in chats:
            [clip] = [
                part
                for message in messages
                for part in message["content"]
                if part["type"] == "video"
            ]
            frames = numpy.stack(
                [
                    numpy.asarray(Image.open(path).convert("RGB"))
                    for path in sorted(Path(clip["video"]).glob("*.png"))
                ]
            )
            prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            batch = processor(
                text=[prompt],
                videos=[[frames]],
                video_metadata=[
                    {
                        "total_num_frames": len(frames),
                        "fps": clip["fps"],
                        "duration": len(frames) / clip["fps"],
                    }
                ],
                do_sample_frames=True,
                return_tensors="pt",
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
verification.check("question", visual_question_answering)
verification.check("image", image_text_to_text)
verification.check("clip", video_text_to_text)
verification.finish()
