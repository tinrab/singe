"""Wav2Vec2, transcribing a clip by CTC with `transformers`' automatic-speech-recognition pipeline, as `mod.rs` transcribes it in Singe."""

from transformers import pipeline

from support import DEVICE, Verification, read_wav


def automatic_speech_recognition(snapshot, dtype):
    recognize = pipeline(
        "automatic-speech-recognition",
        model=snapshot,
        dtype=dtype,
        device=DEVICE,
    )

    def run(paths, parameters):
        outputs = []
        for path in paths:
            samples, rate = read_wav(path)
            outputs.append(
                {
                    "text": recognize(
                        {"raw": samples, "sampling_rate": rate}, **parameters
                    )["text"]
                }
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("transcription", automatic_speech_recognition)
verification.finish()
