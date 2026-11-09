"""Generate module checkpoints, ONNX models, and PyTorch references with uv run --locked.

Run with --check to prove checked-in fixtures are byte-for-byte reproducible.
"""

import argparse
import json
import tempfile
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from ..format import Tensors, write

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "singe/tests/references/layers"
VERSION = "2.12.0"


# The tensors the suite being written has named, which its metadata holds by name.
TENSORS = Tensors()


def data(tensor):
    return TENSORS(tensor)


def weights(shape, offset=0):
    count = 1
    for dimension in shape:
        count *= dimension
    return ((torch.arange(count, dtype=torch.float32) + offset) % 11 - 5).reshape(
        shape
    ) / 16


class FunctionalModule(torch.nn.Module):
    def __init__(self, forward):
        super().__init__()
        self.function = forward

    def forward(self, x):
        return self.function(x)


def encode_model(path, forward, example, opset=18, *, dynamic_shapes=None):
    module = FunctionalModule(forward).eval()
    program = torch.onnx.export(
        module,
        (example,),
        dynamo=True,
        opset_version=opset,
        input_names=["x"],
        output_names=["y"],
        external_data=False,
        optimize=False,
        dynamic_shapes=dynamic_shapes,
    )
    program.save(path, external_data=False)
    model = onnx.load(path)

    # Exporter diagnostics embed checkout paths, Python line numbers, and local
    # build tags. Keep the exported computation and normalize only diagnostics.
    def strip_diagnostics(message):
        for field, value in list(message.ListFields()):
            if field.name in {"metadata_props", "doc_string"}:
                message.ClearField(field.name)
            elif field.message_type is not None:
                for child in value if field.is_repeated else [value]:
                    strip_diagnostics(child)

    strip_diagnostics(model)
    model.producer_version = VERSION
    serialized = model.SerializeToString(deterministic=True)
    assert str(ROOT).encode() not in serialized, "export leaked the checkout path"
    path.write_bytes(serialized)
    onnx.checker.check_model(model)
    return model


def export_model(
    path,
    forward,
    example,
    opset=18,
    *,
    dynamic_shapes=None,
    validation_inputs=(),
    atol=2e-5,
    rtol=2e-5,
):
    model = encode_model(path, forward, example, opset, dynamic_shapes=dynamic_shapes)
    # Validate the complete exporter result, including lifted state and constants.
    from onnx.reference import ReferenceEvaluator

    evaluator = ReferenceEvaluator(model)
    for inputs in (example, *validation_inputs):
        actual = evaluator.run(None, {"x": inputs.numpy()})[0]
        torch.testing.assert_close(
            torch.from_numpy(actual), forward(inputs), atol=atol, rtol=rtol
        )


def generate(destination):
    assert torch.__version__.split("+")[0] == VERSION, torch.__version__
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    destination.mkdir(parents=True, exist_ok=True)
    TENSORS.named.clear()
    fixtures = {}

    state = {
        "first.weight": torch.tensor([[1.0, 2.0, 0.5], [-1.0, 1.0, 2.0]]),
        "first.bias": torch.tensor([1.0, -2.0]),
        "second.weight": torch.tensor([[1.0, 0.25], [-0.5, 1.0]]),
        "second.bias": torch.tensor([0.5, -1.0]),
    }
    inputs = [
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]),
    ]
    forward = lambda x: F.linear(
        F.relu(F.linear(x, state["first.weight"], state["first.bias"])),
        state["second.weight"],
        state["second.bias"],
    )
    export_model(destination / "mlp.onnx", forward, inputs[0])
    save_file(state, destination / "mlp.safetensors")
    fixtures["mlp"] = {
        "state": {name: data(value) for name, value in state.items()},
        "cases": [{"input": data(x), "output": data(forward(x))} for x in inputs],
    }

    state = {
        "conv.weight": torch.tensor(
            [[[[0.25, -0.125], [0.5, 0.25]]], [[[0.125, 0.25], [-0.25, 0.5]]]]
        ),
        "conv.bias": weights((2,), 2),
        "norm.weight": torch.tensor([1.25, 0.75]),
        "norm.bias": torch.tensor([0.1, -0.2]),
        "norm.running_mean": torch.tensor([0.2, -0.1]),
        "norm.running_var": torch.tensor([0.75, 1.5]),
        "head.weight": weights((3, 8), 3),
        "head.bias": weights((3,), 5),
    }
    inputs = [
        torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 8,
        torch.arange(16, dtype=torch.float32).flip(0).reshape(1, 1, 4, 4) / 4 - 1,
    ]

    def forward(x):
        x = F.conv2d(x, state["conv.weight"], state["conv.bias"])
        x = F.batch_norm(
            x,
            state["norm.running_mean"],
            state["norm.running_var"],
            state["norm.weight"],
            state["norm.bias"],
            training=False,
            eps=1e-5,
        )
        return F.linear(
            F.avg_pool2d(F.relu(x), 2, stride=1).flatten(1),
            state["head.weight"],
            state["head.bias"],
        )

    export_model(destination / "cnn.onnx", forward, inputs[0])
    save_file(state, destination / "cnn.safetensors")
    fixtures["cnn"] = {
        "state": {name: data(value) for name, value in state.items()},
        "cases": [{"input": data(x), "output": data(forward(x))} for x in inputs],
    }

    state = {}
    for offset, (name, out_width, in_width) in enumerate(
        [
            ("query", 4, 4),
            ("key", 4, 4),
            ("value", 4, 4),
            ("output", 4, 4),
            ("expand", 6, 4),
            ("contract", 4, 6),
        ]
    ):
        state[name + ".weight"] = weights((out_width, in_width), offset)
        state[name + ".bias"] = weights((out_width,), offset + 3)
    for name in ["norm1", "norm2"]:
        state[name + ".weight"] = torch.tensor([1.0, 0.75, 1.25, 0.5])
        state[name + ".bias"] = weights((4,), 2)
    inputs = [weights((1, 3, 4), 2) * 4, weights((1, 3, 4), 5) * 3]

    def linear(x, name):
        return F.linear(x, state[name + ".weight"], state[name + ".bias"])

    def norm(x, name):
        return F.layer_norm(
            x, [4], state[name + ".weight"], state[name + ".bias"], eps=1e-5
        )

    def forward(x):
        q, k, v = [
            linear(x, name).reshape(1, 3, 2, 2).permute(0, 2, 1, 3)
            for name in ["query", "key", "value"]
        ]
        attention = ((q @ k.transpose(-2, -1)) / (2**0.5)).softmax(-1) @ v
        hidden = norm(
            x + linear(attention.permute(0, 2, 1, 3).reshape(1, 3, 4), "output"),
            "norm1",
        )
        return norm(
            hidden
            + linear(F.gelu(linear(hidden, "expand"), approximate="tanh"), "contract"),
            "norm2",
        )

    export_model(destination / "encoder.onnx", forward, inputs[0], opset=20)
    save_file(state, destination / "encoder.safetensors")
    fixtures["encoder"] = {
        "state": {name: data(value) for name, value in state.items()},
        "cases": [{"input": data(x), "output": data(forward(x))} for x in inputs],
    }
    for name, fixture in fixtures.items():
        outputs = [TENSORS.named[case["output"]["tensor"]] for case in fixture["cases"]]
        assert not torch.equal(outputs[0], outputs[1]), (
            f"{name} fixture does not exercise changed inputs"
        )
    write(
        destination,
        dict(TENSORS.named),
        {
            "reference": {
                "framework": "pytorch",
                "version": VERSION,
                "device": "cpu",
                "deterministic": True,
            },
            "models": fixtures,
        },
    )


def tiny(output):
    output.mkdir(parents=True, exist_ok=True)
    generate(output)
