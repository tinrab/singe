"""CLAP, classifying a clip among candidate labels with `transformers`' zero-shot audio classification pipeline, as `mod.rs` classifies it in Singe."""

from transformers import pipeline

from support import DEVICE, Verification, read_wav


def zero_shot_audio_classification(snapshot, dtype):
    classify = pipeline(
        "zero-shot-audio-classification",
        model=snapshot,
        dtype=dtype,
        device=DEVICE,
    )

    def run(inputs, parameters):
        outputs = []
        for item in inputs:
            samples, _ = read_wav(item["audio"])
            outputs.append(
                classify(
                    samples, candidate_labels=item["candidate_labels"], **parameters
                )
            )
        return outputs

    return run


verification = Verification(__file__)
verification.check("sounds", zero_shot_audio_classification)
verification.finish()
