"""CSM, speaking a sentence greedily with `transformers`' text-to-audio pipeline, and once after a turn whose clip gives the voice, through the processor's chat template and `generate`, the backbone and the depth decoder both greedy, as `mod.rs` speaks them in Singe.
The weights are `unsloth/csm-1b`, which mirrors the gated `sesame/csm-1b`.
In F32 the model is 6.6 GB with its codec, so the reference runs on the host; in BF16 greedy codes part within a few frames on either side, which no waveform comparison survives."""

import torch
from transformers import AutoProcessor, CsmForConditionalGeneration, pipeline

from support import Verification


def text_to_audio(snapshot, dtype):
    speak = pipeline("text-to-audio", model=snapshot, dtype=dtype, device="cpu")

    def run(texts, parameters):
        outputs = []
        for text in texts:
            with torch.no_grad():
                spoken = speak(
                    text,
                    generate_kwargs={
                        "do_sample": False,
                        "depth_decoder_do_sample": False,
                        "max_new_tokens": parameters["max_new_tokens"],
                    },
                )
            outputs.append(
                {
                    "audio": torch.as_tensor(spoken["audio"])
                    .flatten()
                    .float()
                    .tolist(),
                    "sampling_rate": spoken["sampling_rate"],
                }
            )
        return outputs

    return run


def voiced(snapshot, dtype):
    processor = AutoProcessor.from_pretrained(snapshot)
    model = CsmForConditionalGeneration.from_pretrained(snapshot, dtype=dtype)

    def run(conversations, parameters):
        outputs = []
        for conversation in conversations:
            # Singe names a turn's clip in `audio`, where the chat template reads `path`.
            turns = [
                {
                    "role": turn["role"],
                    "content": [
                        {"type": "audio", "path": part["audio"]}
                        if part["type"] == "audio"
                        else part
                        for part in turn["content"]
                    ],
                }
                for turn in conversation
            ]
            inputs = processor.apply_chat_template(
                turns, tokenize=True, return_dict=True
            )
            with torch.no_grad():
                audio = model.generate(
                    **inputs,
                    output_audio=True,
                    do_sample=False,
                    depth_decoder_do_sample=False,
                    max_new_tokens=parameters["max_new_tokens"],
                )[0]
            outputs.append(
                {
                    "audio": audio.flatten().float().tolist(),
                    "sampling_rate": processor.feature_extractor.sampling_rate,
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("speech", text_to_audio)
verification.check("voice", voiced)
verification.finish()
