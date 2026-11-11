"""Every suite, by name: each a module with `OUTPUT` and `tiny(output)`.

Most suites are PyTorch's references; `onnx_schemas` and `model_operations` write what Singe generates from ONNX and `transformers`, which the same check keeps current.
"""

from . import (
    blocks,
    conformance,
    layers,
    model_operations,
    onnx_conformance,
    onnx_schemas,
    set_encoder,
    shapes,
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
    ]
}
