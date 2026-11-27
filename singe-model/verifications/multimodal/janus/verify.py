"""Janus-Pro-1B, describing an image in BF16 with its processor and `generate`, and generating an image in F32 with `generate(generation_mode="image")` and the processor's post-processing, as `mod.rs` does both in Singe.
The whole model is 8.3 GB in F32, so describing runs in BF16; generating reads only the decoder, the generation heads, and the VQ decoder, 6.1 GB, so the reference drops the rest before moving to the GPU.
In BF16 the guided codes part on the second of 576 steps: classifier-free guidance at scale 5 takes `5c − 4u` of the two rows' logits, which magnifies their rounding past the 0.375 between the top two codes.
`hf-tiny-v2/tiny-random-JanusForConditionalGeneration` cannot stand in: its generation aligner gives 10 features to a decoder 32 wide, which `transformers` fails on too, and without a processor config its processor expands an image to 576 tokens for a tower that gives 16.
Janus-Pro's generation config names the image start the unguided row keeps; the reference sets it on the model's config, which a checkpoint without it would need."""

import torch
from transformers import AutoProcessor, JanusForConditionalGeneration

from support import DEVICE, Verification, data_url, generation


def image_text_to_text(snapshot, dtype):
    processor = AutoProcessor.from_pretrained(snapshot)
    model = (
        JanusForConditionalGeneration.from_pretrained(snapshot, dtype=dtype)
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
            batch["pixel_values"] = batch["pixel_values"].to(dtype)
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


def text_to_image(snapshot, dtype):
    processor = AutoProcessor.from_pretrained(snapshot)
    model = JanusForConditionalGeneration.from_pretrained(snapshot, dtype=dtype).eval()
    # Image generation reads none of these, and without them the F32 model fits 8 GB.
    # `generate` reads the vision tower's config for the image's code count, so its layers go and it stays.
    model.model.vision_model.encoder.layers = torch.nn.ModuleList()
    model.model.aligner = None
    model.model.vqmodel.encoder = None
    model.lm_head = None
    model = model.to(DEVICE)
    start = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.boi_token)
    model.generation_config.generation_kwargs = {"boi_token_id": start}

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": item["prompt"]}]}
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            batch = processor(
                text=prompt, generation_mode="image", return_tensors="pt"
            ).to(DEVICE)
            with torch.no_grad():
                codes = model.generate(
                    **batch, generation_mode="image", do_sample=False
                )
                decoded = model.decode_image_tokens(codes)
            # `postprocess` gives each image `[3, H, W]` as `uint8`; it refuses to give PIL images here.
            [image] = processor.postprocess(list(decoded.float()))["pixel_values"]
            outputs.append(data_url(image.permute(1, 2, 0).cpu().numpy()))
        return outputs

    return run


verification = Verification(__file__)
verification.check("image", image_text_to_text)
verification.check("generate", text_to_image)
verification.finish()
