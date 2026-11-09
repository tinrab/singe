"""Generate a real dynamic-batch/sequence encoder export and pinned references.

Run with just references shapes --check.
The checkpoint is the existing module encoder's immutable state.
"""

import argparse
import json
import tempfile
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from ..format import write
from .layers import ROOT, TENSORS, VERSION, data, export_model, weights

OUTPUT = ROOT / "singe/tests/references/shapes"


def generate_shape_cases(destination, inputs):
    from onnx import TensorProto, helper
    from onnx.reference import ReferenceEvaluator

    def integer(name, dimensions, values):
        return helper.make_tensor(name, TensorProto.INT64, dimensions, values)

    models = {
        "shape_as_data": (
            [
                helper.make_node("Shape", ["x"], ["shape"]),
                helper.make_node("Add", ["shape", "one"], ["y"]),
            ],
            [integer("one", [], [1])],
            TensorProto.INT64,
            [3],
        ),
        "scalar_shape_gather": (
            [
                helper.make_node("Shape", ["x"], ["shape"]),
                helper.make_node("Gather", ["shape", "axis"], ["y"], axis=0),
            ],
            [integer("axis", [], [0])],
            TensorProto.INT64,
            [],
        ),
        "signed_shape_division": (
            [
                helper.make_node("Shape", ["x"], ["shape"]),
                helper.make_node("Gather", ["shape", "axis"], ["batch"], axis=0),
                helper.make_node("Mul", ["batch", "negative"], ["negative_batch"]),
                helper.make_node("Div", ["negative_batch", "two"], ["y"]),
            ],
            [
                integer("axis", [1], [0]),
                integer("negative", [], [-1]),
                integer("two", [], [2]),
            ],
            TensorProto.INT64,
            [1],
        ),
        "reshape_gathered_shape": (
            [
                helper.make_node("Shape", ["x"], ["shape"]),
                helper.make_node("Gather", ["shape", "order"], ["target"], axis=0),
                helper.make_node("Reshape", ["x", "target"], ["y"], allowzero=1),
            ],
            [integer("order", [3], [1, 0, 2])],
            TensorProto.FLOAT,
            ["sequence", "batch", 4],
        ),
    }
    references = {}
    for name, (nodes, initializers, dtype, output_shape) in models.items():
        graph = helper.make_graph(
            nodes,
            name,
            [
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, ["batch", "sequence", 4]
                )
            ],
            [helper.make_tensor_value_info("y", dtype, output_shape)],
            initializers,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=10
        )
        onnx.checker.check_model(model)
        (destination / (name + ".onnx")).write_bytes(
            model.SerializeToString(deterministic=True)
        )
        evaluator = ReferenceEvaluator(model)
        cases = []
        for x in inputs:
            output = torch.from_numpy(evaluator.run(None, {"x": x.numpy()})[0])
            cases.append({"input": data(x), "output": data(output)})
        references[name] = {
            "dtype": "i64" if dtype == TensorProto.INT64 else "f32",
            "cases": cases,
        }
    write(destination / "shape_cases", dict(TENSORS.named), references)
    TENSORS.named.clear()

    # Boundary fixtures are checker-valid and executable at their representative
    # shapes. Rust tests exercise annotations and deliberately unsupported shape
    # sources at the import/preparation boundary.
    def boundary(
        name,
        nodes,
        *,
        extra_inputs=(),
        initializers=(),
        value_info=(),
        output_shape=("batch", "sequence", 4),
        output_type=TensorProto.FLOAT,
        sample=None,
        feeds=None,
        input_shape=("batch", "sequence", 4),
    ):
        graph = helper.make_graph(
            nodes,
            name,
            [
                helper.make_tensor_value_info(
                    "x", TensorProto.FLOAT, list(input_shape)
                ),
                *extra_inputs,
            ],
            [helper.make_tensor_value_info("y", output_type, list(output_shape))],
            list(initializers),
            value_info=list(value_info),
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 20)], ir_version=10
        )
        onnx.checker.check_model(model)
        (destination / (name + ".onnx")).write_bytes(
            model.SerializeToString(deterministic=True)
        )
        ReferenceEvaluator(model).run(
            None,
            {"x": (inputs[0] if sample is None else sample).numpy(), **(feeds or {})},
        )

    boundary(
        "fixed_output_annotation",
        [helper.make_node("Identity", ["x"], ["y"])],
        output_shape=(2, "sequence", 4),
    )
    boundary(
        "fixed_intermediate_annotation",
        [
            helper.make_node("Relu", ["x"], ["hidden"]),
            helper.make_node("Identity", ["hidden"], ["y"]),
        ],
        value_info=[
            helper.make_tensor_value_info(
                "hidden", TensorProto.FLOAT, [2, "sequence", 4]
            )
        ],
    )
    boundary(
        "data_dependent_reshape",
        [helper.make_node("Reshape", ["x", "target"], ["y"], allowzero=1)],
        extra_inputs=[helper.make_tensor_value_info("target", TensorProto.INT64, [3])],
        feeds={"target": torch.tensor([2, 3, 4]).numpy()},
    )
    boundary(
        "unsupported_float_shape",
        [
            helper.make_node("Shape", ["x"], ["shape"]),
            helper.make_node("Cast", ["shape"], ["floating"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["floating"], ["target"], to=TensorProto.INT64),
            helper.make_node("Reshape", ["x", "target"], ["y"], allowzero=1),
        ],
    )
    boundary(
        "static_metadata_expand",
        [
            helper.make_node("Shape", ["x"], ["target"]),
            helper.make_node("Expand", ["one", "target"], ["y"]),
        ],
        initializers=[helper.make_tensor("one", TensorProto.FLOAT, [], [1.0])],
        input_shape=(2, 3, 4),
        output_shape=(2, 3, 4),
    )
    boundary(
        "shape_division_by_zero",
        [
            helper.make_node("Shape", ["x"], ["shape"]),
            helper.make_node("Gather", ["shape", "axis"], ["batch"], axis=0),
            helper.make_node("Sub", ["batch", "one"], ["denominator"]),
            helper.make_node("Div", ["batch", "denominator"], ["y"]),
        ],
        initializers=[integer("axis", [], [0]), integer("one", [], [1])],
        output_type=TensorProto.INT64,
        output_shape=(),
    )
    boundary(
        "shape_overflow",
        [
            helper.make_node("Shape", ["x"], ["shape"]),
            helper.make_node("Gather", ["shape", "axis"], ["batch"], axis=0),
            helper.make_node("Mul", ["batch", "factor"], ["y"]),
        ],
        initializers=[integer("axis", [], [0]), integer("factor", [], [2**63 - 1])],
        output_type=TensorProto.INT64,
        output_shape=(),
        sample=torch.empty((0, 1, 4)),
    )


def generate(destination):
    assert torch.__version__.split("+")[0] == VERSION, torch.__version__
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    destination.mkdir(parents=True, exist_ok=True)
    TENSORS.named.clear()
    state = load_file(ROOT / "singe/tests/references/layers/encoder.safetensors")

    def linear(x, name):
        return F.linear(x, state[name + ".weight"], state[name + ".bias"])

    def norm(x, name):
        return F.layer_norm(
            x, [4], state[name + ".weight"], state[name + ".bias"], eps=1e-5
        )

    def forward(x):
        batch, sequence, width = x.shape
        heads = 2
        head_width = width // heads
        # A common BMM attention layout. The merged batch/head dimension must be
        # evaluated from metadata for each runtime shape, never tensor contents.
        q, k, v = [
            linear(x, name)
            .reshape(batch, sequence, heads, head_width)
            .permute(0, 2, 1, 3)
            .reshape(batch * heads, sequence, head_width)
            for name in ["query", "key", "value"]
        ]
        attention = ((q @ k.transpose(-2, -1)) / (head_width**0.5)).softmax(-1) @ v
        attention = (
            attention.reshape(batch, heads, sequence, head_width)
            .permute(0, 2, 1, 3)
            .reshape(batch, sequence, width)
        )
        hidden = norm(x + linear(attention, "output"), "norm1")
        return norm(
            hidden
            + linear(F.gelu(linear(hidden, "expand"), approximate="tanh"), "contract"),
            "norm2",
        )

    sizes = [(2, 3), (1, 1), (3, 5), (4, 8), (2, 3)]
    inputs = [
        weights((batch, sequence, 4), offset + 2) * 3
        for offset, (batch, sequence) in enumerate(sizes)
    ]
    path = destination / "encoder.onnx"
    export_model(
        path,
        forward,
        inputs[0],
        opset=20,
        dynamic_shapes={
            "x": {
                0: torch.export.Dim("batch", min=1, max=4),
                1: torch.export.Dim("sequence", min=1, max=8),
            }
        },
        validation_inputs=inputs[1:],
    )
    model = onnx.load(path)
    dimensions = model.graph.input[0].type.tensor_type.shape.dim
    assert [dimension.dim_param for dimension in dimensions[:2]] == [
        "batch",
        "sequence",
    ]
    operators = sorted({node.op_type for node in model.graph.node})
    assert {"Shape", "Mul", "Concat", "Reshape"} <= set(operators), operators
    references = {
        "reference": {
            "framework": "pytorch",
            "version": VERSION,
            "device": "cpu",
            "deterministic": True,
        },
        "checkpoint": "../modules/encoder.safetensors",
        "constraints": {"batch": [1, 4], "sequence": [1, 8], "width": 4},
        "operators": operators,
        "cases": [{"input": data(x), "output": data(forward(x))} for x in inputs],
    }
    write(destination, dict(TENSORS.named), references)
    TENSORS.named.clear()
    generate_shape_cases(destination, inputs)


def tiny(output):
    output.mkdir(parents=True, exist_ok=True)
    generate(output)
