"""Silero VAD v5, each 32 ms window's probability of speech over a clip and over a clip with pauses, from its ONNX file in onnxruntime, window by window with the state carried, as `mod.rs` computes it in Singe.

Each window is 512 samples at 16 kHz after the last 64 samples of the window before, zeros before the first, as the `silero-vad` package's `OnnxWrapper` feeds them; the clip's last partial window is left out.
"""

from pathlib import Path

import numpy

from support import Verification, read_audio

WINDOW = 512
CONTEXT = 64


def voice_activity(snapshot, dtype):
    import onnxruntime

    session = onnxruntime.InferenceSession(
        str(Path(snapshot) / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"]
    )

    def run(inputs, parameters):
        outputs = []
        for path in inputs:
            samples, rate = read_audio(path)
            assert rate == 16000, f"{path} is at {rate} Hz, not 16 kHz"
            state = numpy.zeros((2, 1, 128), dtype=numpy.float32)
            context = numpy.zeros(CONTEXT, dtype=numpy.float32)
            probabilities = []
            for start in range(0, len(samples) - WINDOW + 1, WINDOW):
                window = samples[start : start + WINDOW].astype(numpy.float32)
                feed = numpy.concatenate([context, window])[None, :]
                probability, state = session.run(
                    None,
                    {
                        "input": feed,
                        "state": state,
                        "sr": numpy.array(16000, dtype=numpy.int64),
                    },
                )
                context = window[-CONTEXT:]
                probabilities.append(float(probability[0, 0]))
            outputs.append(probabilities)
        return outputs

    return run


verification = Verification(__file__)
verification.check("speech", voice_activity)
verification.check("pauses", voice_activity)
verification.finish()
