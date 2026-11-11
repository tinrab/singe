"""Every suite, by name: each a module with `OUTPUT`, and `tiny(output)`, `real(output)`, or both.

Most suites are PyTorch's references; `onnx_schemas` and `model_operations` write what Singe generates from ONNX and `transformers`, which the same check keeps current.
"""

from . import (
    bert,
    blocks,
    conformance,
    deepseek_v3,
    dinov2,
    gemma2,
    gpt2,
    layers,
    llama,
    model_operations,
    onnx_conformance,
    onnx_schemas,
    qwen3,
    qwen3_5,
    qwen3_asr,
    set_encoder,
    shapes,
    taesd,
    tokenization,
    whisper,
)

SUITES = {
    module.__name__.rsplit(".", 1)[1]: module
    for module in [
        conformance,
        onnx_schemas,
        onnx_conformance,
        model_operations,
        layers,
        shapes,
        set_encoder,
        blocks,
        tokenization,
        qwen3,
        qwen3_5,
        llama,
        gemma2,
        gpt2,
        deepseek_v3,
        bert,
        taesd,
        dinov2,
        whisper,
        qwen3_asr,
    ]
}
