"""MMS-TTS, a VITS model, speaking text with `transformers`' text-to-speech pipeline, its noise scales set on the model as the case gives them, since torch's random draws cannot be repeated in Singe, as `mod.rs` speaks it."""

from transformers import pipeline

from support import DEVICE, Verification


def text_to_audio(snapshot, dtype):
    speak = pipeline("text-to-speech", model=snapshot, dtype=dtype, device=DEVICE)

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
