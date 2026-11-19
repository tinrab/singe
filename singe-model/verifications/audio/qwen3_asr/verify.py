"""Qwen3-ASR, transcribing a clip and answering an instruction about one greedily with `transformers`' `generate`, as `mod.rs` does both in Singe."""

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from support import DEVICE, Verification, read_wav


def automatic_speech_recognition(snapshot, dtype):
    processor = AutoProcessor.from_pretrained(snapshot)
    model = (
        AutoModelForMultimodalLM.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )

    def run(paths, parameters):
        outputs = []
        for path in paths:
            samples, _ = read_wav(path)
            request = processor.apply_transcription_request(audio=samples).to(DEVICE)
            with torch.no_grad():
                generated = model.generate(
                    **request,
                    max_new_tokens=parameters["max_new_tokens"],
                    num_beams=1,
                    do_sample=False,
                )
            text = processor.decode(
                generated[0, request.input_ids.shape[1] :], skip_special_tokens=True
            )
            # The answer names the clip's language, then gives the text after `<asr_text>`.
            outputs.append({"text": text.split("<asr_text>")[-1]})
        return outputs

    return run


verification = Verification(__file__)
verification.check("transcription", automatic_speech_recognition)


def audio_text_to_text(snapshot, dtype):
    processor = AutoProcessor.from_pretrained(snapshot)
    model = (
        AutoModelForMultimodalLM.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            samples, _ = read_wav(item["audio"])
            # The instruction in the system's turn, where the chat template takes text; the user's turn holds the clip.
            messages = [
                {"role": "system", "content": [{"type": "text", "text": item["text"]}]},
                {"role": "user", "content": [{"type": "audio", "audio": samples}]},
            ]
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
                    num_beams=1,
                    do_sample=False,
                )
            outputs.append(
                {
                    "generated_text": processor.decode(
                        generated[0, batch["input_ids"].shape[1] :],
                        skip_special_tokens=True,
                    )
                }
            )
        return outputs

    return run


verification.check("instructed", audio_text_to_text)
verification.finish()
