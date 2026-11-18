"""Whisper, transcribing a clip greedily in English with `transformers`' `generate`, as `verify.rs` transcribes it in Singe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import AutoProcessor, WhisperForConditionalGeneration  # noqa: E402

from support import DEVICE, Verification, read_wav  # noqa: E402


def automatic_speech_recognition(snapshot):
    processor = AutoProcessor.from_pretrained(snapshot)
    model = (
        WhisperForConditionalGeneration.from_pretrained(snapshot, dtype=torch.float32)
        .to(DEVICE)
        .eval()
    )

    def run(paths, parameters):
        outputs = []
        for path in paths:
            samples, rate = read_wav(path)
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
