"""MusicGen, generating a clip from a text greedily, and once sampled from the likeliest code alone, `top_k` 1, which leaves `generate` and Singe no choice to draw, with `transformers`' tokenizer and `MusicgenForConditionalGeneration.generate`, as `mod.rs` generates it in Singe.
Guidance stays at the generation config's scale, and attention is eager, whose fully masked rows give the unguided row's cross-attention the zeros Singe gives it."""

import torch
from transformers import AutoTokenizer, MusicgenForConditionalGeneration

from support import DEVICE, Verification


def text_to_audio(snapshot, dtype):
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    model = (
        MusicgenForConditionalGeneration.from_pretrained(
            snapshot, dtype=dtype, attn_implementation="eager"
        )
        .to(DEVICE)
        .eval()
    )

    def run(texts, parameters):
        outputs = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                audio = model.generate(
                    **inputs,
                    do_sample=parameters["do_sample"],
                    top_k=parameters.get("top_k"),
                    max_new_tokens=parameters["max_new_tokens"],
                )[0, 0]
            outputs.append(
                {
                    "audio": audio.float().cpu().tolist(),
                    "sampling_rate": model.config.audio_encoder.sampling_rate,
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("music", text_to_audio)
verification.check("sampled", text_to_audio)
verification.finish()
