"""Whisper, transcribing a clip greedily in English with `transformers`' `generate`, as `mod.rs` transcribes it in Singe."""

import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

from support import DEVICE, Verification, read_audio


def automatic_speech_recognition(snapshot, dtype):
    processor = AutoProcessor.from_pretrained(snapshot)
    model = (
        WhisperForConditionalGeneration.from_pretrained(snapshot, dtype=dtype)
        .to(DEVICE)
        .eval()
    )

    def run(paths, parameters):
        outputs = []
        for path in paths:
            samples, rate = read_audio(path)
            features = processor(
                samples, sampling_rate=rate, return_tensors="pt"
            ).input_features.to(DEVICE)
            with torch.no_grad():
                # Singe decodes greedily, which the snapshot's generation config may not ask for.
                generated = model.generate(
                    features,
                    language="en",
                    task="transcribe",
                    max_new_tokens=parameters["max_new_tokens"],
                    num_beams=1,
                    do_sample=False,
                )
            outputs.append(
                {"text": processor.batch_decode(generated, skip_special_tokens=True)[0]}
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("transcription", automatic_speech_recognition)
verification.finish()
