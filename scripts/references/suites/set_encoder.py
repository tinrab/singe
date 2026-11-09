"""Generate the categorical set encoder's checkpoints, exports, and references.

Run with just references set_encoder --check.
The model consumes unpadded categorical IDs, embeds each item, applies the same
encoder block as the earlier fixtures, pools the set, and classifies it. It has
no positional features, padding masks, dropout, or mutable state.
"""

import argparse
import json
import tempfile
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ..format import write
from .layers import (
    ROOT,
    TENSORS,
    VERSION,
    data,
    encode_model,
    export_model,
    weights,
)

OUTPUT = ROOT / "singe/tests/references/set_encoder"
DTYPES = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
TOLERANCES = {"f32": (2e-5, 2e-5), "f16": (3e-3, 3e-3), "bf16": (3e-2, 3e-2)}


def forward(state, x):
    def linear(value, name):
        return F.linear(value, state[name + ".weight"], state[name + ".bias"])

    def norm(value, name):
        return F.layer_norm(
            value, [4], state[name + ".weight"], state[name + ".bias"], eps=1e-5
        )

    hidden = F.embedding(x, state["embedding.weight"])
    batch, sequence, width = hidden.shape
    q, k, v = [
        linear(hidden, name)
        .reshape(batch, sequence, 2, 2)
        .permute(0, 2, 1, 3)
        .reshape(batch * 2, sequence, 2)
        for name in ["query", "key", "value"]
    ]
    attended = ((q @ k.transpose(-2, -1)) / (2**0.5)).softmax(-1) @ v
    attended = (
        attended.reshape(batch, 2, sequence, 2)
        .permute(0, 2, 1, 3)
        .reshape(batch, sequence, width)
    )
    hidden = norm(hidden + linear(attended, "output"), "norm1")
    hidden = norm(
        hidden
        + linear(F.gelu(linear(hidden, "expand"), approximate="tanh"), "contract"),
        "norm2",
    )
    return linear(hidden.mean(1), "head").float()


def generate(destination):
    assert torch.__version__.split("+")[0] == VERSION, torch.__version__
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    destination.mkdir(parents=True, exist_ok=True)
    TENSORS.named.clear()
    original = load_file(ROOT / "singe/tests/references/layers/encoder.safetensors")
    original.update(
        {
            "embedding.weight": weights((32, 4), 7),
            "head.weight": weights((3, 4), 9),
            "head.bias": weights((3,), 4),
        }
    )
    inputs = [
        ((torch.arange(batch * sequence) * 7 + offset * 11) % 32).reshape(
            batch, sequence
        )
        for offset, (batch, sequence) in enumerate(
            [(2, 3), (1, 1), (3, 5), (4, 8), (2, 3)]
        )
    ]
    fixtures = {
        "reference": {
            "framework": "pytorch",
            "version": VERSION,
            "device": "cpu",
            "deterministic": True,
        },
        "constraints": {"batch": [1, 4], "sequence": [1, 8], "vocabulary": 32},
        "models": {},
    }
    for name, dtype in DTYPES.items():
        state = {key: value.to(dtype) for key, value in original.items()}
        fp32 = {key: value.float() for key, value in state.items()}
        checkpoint = name + ".safetensors"
        save_file(state, destination / checkpoint)
        atol, rtol = TOLERANCES[name]
        model_path = destination / (name + ".onnx")
        dimensions = {
            "x": {
                0: torch.export.Dim("batch", min=1, max=4),
                1: torch.export.Dim("sequence", min=1, max=8),
            }
        }
        if name == "bf16":
            # The pinned NumPy/ReferenceEvaluator BF16 MatMul returns FP32 and
            # raises RuntimeTypeError. BF16 uses checker and PyTorch references;
            # its eventual Singe CUDA execution must be tested separately.
            encode_model(
                model_path,
                lambda x: forward(state, x),
                inputs[0],
                opset=20,
                dynamic_shapes=dimensions,
            )
        else:
            export_model(
                model_path,
                lambda x: forward(state, x),
                inputs[0],
                opset=20,
                dynamic_shapes=dimensions,
                validation_inputs=inputs[1:],
                atol=atol,
                rtol=rtol,
            )
        model = onnx.load(model_path)
        operators = sorted({node.op_type for node in model.graph.node})
        assert {"Gather", "ReduceMean", "MatMul", "LayerNormalization"} <= set(
            operators
        ), operators
        if name != "f32":
            assert "Cast" in operators, operators
        cases = []
        for x in inputs:
            actual = forward(state, x)
            reference = forward(fp32, x)
            torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
            cases.append(
                {
                    "input": data(x),
                    "output": data(actual),
                    "fp32_accumulation_reference": data(reference),
                }
            )
        fixtures["models"][name] = {
            "checkpoint": checkpoint,
            "onnx_reference_evaluator": name != "bf16",
            "atol": atol,
            "rtol": rtol,
            "operators": operators,
            "cases": cases,
        }
    write(destination, dict(TENSORS.named), fixtures)


def tiny(output):
    output.mkdir(parents=True, exist_ok=True)
    generate(output)
