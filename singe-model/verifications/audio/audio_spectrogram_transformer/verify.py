"""The Audio Spectrogram Transformer, classifying a clip with `transformers`' audio-classification pipeline, as `mod.rs` classifies it in Singe."""

from transformers import pipeline

from support import DEVICE, Verification, read_audio


def audio_classification(snapshot, dtype):
    classify = pipeline(
        "audio-classification", model=snapshot, dtype=dtype, device=DEVICE
    )

    def run(paths, parameters):
        outputs = []
        for path in paths:
            samples, rate = read_audio(path)
            outputs.append(
                classify({"raw": samples, "sampling_rate": rate}, **parameters)
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("speech", audio_classification)
verification.finish()
