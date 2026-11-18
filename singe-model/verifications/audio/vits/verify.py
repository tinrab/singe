"""MMS-TTS, a VITS model, speaking text with `transformers`' text-to-speech pipeline, its noise scales set on the model as the case gives them, since torch's random draws cannot be repeated in Singe, as `verify.rs` speaks it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from support import DEVICE, Verification  # noqa: E402


def text_to_audio(snapshot):
    speak = pipeline(
        "text-to-speech", model=snapshot, dtype=torch.float32, device=DEVICE
    )

    def run(texts, parameters):
        speak.model.noise_scale = parameters["noise_scale"]
        speak.model.noise_scale_duration = parameters["noise_scale_duration"]
        outputs = []
        for text in texts:
            spoken = speak(text)
            outputs.append(
                {
                    "audio": spoken["audio"].reshape(-1).tolist(),
                    "sampling_rate": spoken["sampling_rate"],
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("speech", text_to_audio)
verification.finish()
