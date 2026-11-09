"""Generate deterministic operator conformance cases from ONNX reference execution.

Use just references onnx_conformance [--check].
Each admitted schema family has a fixture. Model exports are tested separately.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
from onnx.reference import ReferenceEvaluator

from .onnx_schemas import RANGES

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "singe/tests/references/onnx"


def tensor(value):
    value = np.asarray(value)
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "bytes": value.tobytes().hex(),
    }


def fixture(op, version, variant="", constant=False):
    x = np.array([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]], dtype=np.float32)
    runtime = {"x": x}
    initializers = {}
    inputs = ["x"]
    outputs = ["y"]
    attrs = {}
    prefix = []

    def const(name, value, dtype=None):
        initializers[name] = np.asarray(value, dtype=dtype)
        return name

    def extra(name, value, dtype=None):
        inputs.append(const(name, value, dtype))

    if op in {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Max",
        "Min",
        "Equal",
        "Less",
        "LessOrEqual",
        "Greater",
        "GreaterOrEqual",
    }:
        extra("other", np.array([[0.5, 0.2, 0.9]], dtype=np.float32))
    elif op in {"Acosh"}:
        runtime["x"] = x + 1
    elif op == "Pow":
        extra("exponent", 2.0, np.float32)
    elif op in {"Gemm", "MatMul"}:
        extra("weight", [[1.0, 0.5], [-0.5, 1.0], [0.25, -0.25]], np.float32)
        if op == "Gemm":
            extra("bias", [1.0, -0.5], np.float32)
    elif op == "Clip":
        extra("minimum", 0.3, np.float32)
        extra("maximum", 0.6, np.float32)
    elif op == "Where":
        inputs = [
            const("condition", [[True, False, True], [False, True, False]], np.bool_),
            "x",
            const("other", -x),
        ]
    elif op == "Concat":
        extra("other", -x)
        attrs["axis"] = 1
    elif op in {"ArgMax", "ArgMin", "Softmax", "LogSoftmax"}:
        attrs["axis"] = 1
    elif op.startswith("Reduce"):
        if op == "ReduceSum" or version >= 18:
            extra("axes", [1], np.int64)
        else:
            attrs["axes"] = [1]
        attrs["keepdims"] = 0
    elif op == "Reshape":
        extra("shape", [3, 2], np.int64)
    elif op == "Expand":
        runtime["x"] = x[:1]
        extra("shape", [2, 3], np.int64)
    elif op == "Squeeze":
        runtime["x"] = x.reshape(1, 2, 3)
    elif op == "Unsqueeze":
        extra("axes", [0], np.int64)
    elif op == "Slice":
        extra("starts", [0], np.int64)
        extra("ends", [2], np.int64)
        extra("axes", [1], np.int64)
    elif op == "Gather":
        extra("indices", [2, 0], np.int64)
        attrs["axis"] = 1
    elif op == "ScatterElements":
        # Singe imports rank-one scatters, the form its exporter writes for strided scatters.
        runtime["x"] = x.reshape(-1)
        extra("indices", [4, 1], np.int64)
        extra("updates", [9.0, 8.0], np.float32)
        attrs["axis"] = 0
    elif op == "Transpose":
        attrs["perm"] = [1, 0]
    elif op == "Pad":
        extra("pads", [0, 1, 0, 1], np.int64)
    elif op == "Cast":
        attrs["to"] = onnx.TensorProto.INT64
    elif op == "Constant":
        runtime = {}
        inputs = []
        attrs["value_floats"] = [0.25, -0.5, 0.75]
    elif op == "ConstantOfShape":
        runtime = {}
        inputs = [const("shape", [2, 3], np.int64)]
    elif op in {
        "Conv",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "BatchNormalization",
    }:
        runtime["x"] = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) / 4
        if op == "Conv":
            extra("weight", [[[[0.25, -0.5], [0.75, 0.125]]]], np.float32)
            extra("bias", [0.2], np.float32)
        elif op == "BatchNormalization":
            for name, value in [
                ("scale", [1.25]),
                ("bias", [0.2]),
                ("mean", [0.3]),
                ("var", [0.75]),
            ]:
                extra(name, value, np.float32)
        elif op != "GlobalAveragePool":
            attrs["kernel_shape"] = [2, 2]
            if variant == "dilation_one":
                attrs["dilations"] = [1, 1]
    elif op == "LayerNormalization":
        extra("scale", [1.0, 0.75, 1.25], np.float32)
        extra("bias", [0.1, -0.2, 0.3], np.float32)
    elif op == "Gelu":
        attrs["approximate"] = "none" if variant == "exact" else "tanh"
    elif op == "Attention":
        runtime["x"] = np.array([[[[0.2, 0.5], [0.75, -0.25]]]], dtype=np.float32)
        extra("key", runtime["x"] * 0.5)
        extra("value", runtime["x"] * -0.75)
    elif op == "TopK":
        extra("k", [2], np.int64)
        outputs.append("indices")
    elif op in {"QuantizeLinear", "DequantizeLinear"}:
        if op == "DequantizeLinear":
            runtime["x"] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        extra("scale", 0.25, np.float32)
        extra("zero", 0, np.uint8)
    elif op == "QLinearMatMul":
        runtime["x"] = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        inputs = [
            "x",
            const("a_scale", 0.25, np.float32),
            const("a_zero", 0, np.uint8),
            const("b", [[1, 0], [0, 1], [1, 1]], np.uint8),
            const("b_scale", 0.5, np.float32),
            const("b_zero", 0, np.uint8),
            const("y_scale", 0.125, np.float32),
            const("y_zero", 0, np.uint8),
        ]
    if variant == "edges":
        runtime["x"] = np.array(
            [[np.nan, -0.0, 0.0], [-np.inf, np.inf, -1.0]], dtype=np.float32
        )
        if op == "Clip":
            initializers["minimum"] = np.array(-1.0, dtype=np.float32)
            initializers["maximum"] = np.array(1.0, dtype=np.float32)
    elif variant.startswith("halfway_"):
        _, dtype, zero = variant.split("_")
        runtime["x"] = np.array(
            [[-200.0, -2.5, -1.5, -0.5], [0.5, 1.5, 2.5, 300.0]], dtype=np.float32
        )
        initializers["scale"] = np.array(1.0, dtype=np.float32)
        initializers["zero"] = np.array(int(zero), dtype=dtype)
    elif op == "LayerNormalization" and variant:
        dtype, *layout = variant.split("_")
        dtype = {"f32": "float32", "f16": "float16", "bf16": "bfloat16"}[dtype]
        runtime["x"] = (
            np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 8 - 1
        ).astype(dtype)
        width = 3 if layout else 4
        initializers["scale"] = np.asarray([1.25, 0.75, 1.5, 0.5][:width], dtype=dtype)
        initializers["bias"] = np.asarray(
            [0.125, -0.25, 0.5, -0.125][:width], dtype=dtype
        )
        attrs["axis"] = -1
        if layout:
            prefix = [h.make_node("Transpose", ["x"], ["transposed"], perm=[0, 2, 1])]
            inputs[0] = "transposed"
    elif op == "Cast" and variant:
        source, target = variant.split("_")
        runtime["x"] = np.array(
            [[-1.5, 0.0, 1.0004], [1e-6, -0.0, 600.0]],
            dtype={"f32": "float32", "f16": "float16", "bf16": "bfloat16"}[source],
        )
        attrs["to"] = {
            "f32": onnx.TensorProto.FLOAT,
            "f16": onnx.TensorProto.FLOAT16,
            "bf16": onnx.TensorProto.BFLOAT16,
        }[target]
        if version >= 19:
            attrs["saturate"] = 0
    elif op == "Gather" and variant:
        kind, dtype, index_dtype = variant.split("_")
        runtime["x"] = np.arange(24).reshape(2, 3, 4).astype(dtype)
        initializers.pop("indices")
        attrs["axis"] = 1
        indices = np.array([[-1, 0], [1, -2]], dtype=index_dtype)
        if kind == "scalar":
            indices = np.array(-1, dtype=index_dtype)
        elif kind == "empty":
            indices = np.empty((0, 2), dtype=index_dtype)
        elif kind == "zero":
            runtime["x"] = np.empty((2, 0, 4), dtype=dtype)
            indices = np.empty((0, 2), dtype=index_dtype)
        elif kind == "last":
            attrs["axis"] = -1
        elif kind == "bits":
            if dtype == "int64":
                runtime["x"] += 2**60 + 3
            elif dtype == "bool":
                runtime["x"] = (np.arange(24).reshape(2, 3, 4) % 3) == 0
            else:
                runtime["x"].reshape(-1)[:4] = [np.nan, -0.0, np.inf, -np.inf]
        runtime["indices"] = indices
    if constant:
        initializers.update(runtime)
        runtime = {}
    node = h.make_node(op, inputs, outputs, name=f"conformance_{op}_{version}", **attrs)
    infos = [
        h.make_tensor_value_info(
            name, nh.from_array(value).data_type, list(value.shape)
        )
        for name, value in runtime.items()
    ]
    graph = h.make_graph(
        [*prefix, node],
        "conformance",
        infos,
        [
            h.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
            for name in outputs
        ],
        [nh.from_array(value, name) for name, value in initializers.items()],
    )
    model = h.make_model(
        graph, opset_imports=[h.make_opsetid("", version)], ir_version=10
    )
    if op == "BatchNormalization" and version < 14:
        # The pre-14 reference implementation selects training from the default
        # momentum attribute. ONNX schema 9 selects inference by one output.
        # https://onnx.ai/onnx/operators/onnx__BatchNormalization.html#batchnormalization-9
        actual = [
            (runtime["x"] - initializers["mean"].reshape(1, -1, 1, 1))
            / np.sqrt(initializers["var"].reshape(1, -1, 1, 1) + np.float32(1e-5))
            * initializers["scale"].reshape(1, -1, 1, 1)
            + initializers["bias"].reshape(1, -1, 1, 1)
        ]
    elif op == "DequantizeLinear":
        # ONNX 1.21 lacks a pre-19 reference class. This per-tensor uint8
        # case uses the schema's affine formula without precision extensions.
        actual = [
            (runtime["x"].astype(np.float32) - initializers["zero"])
            * initializers["scale"]
        ]
    elif (
        op == "Gather"
        and (runtime.get("indices", initializers.get("indices"))).size == 0
    ):
        # ONNX 1.21 op_gather.py returns shape (0,) for every empty index tensor.
        # NumPy take preserves the specified input-prefix/indices/input-suffix
        # dimensions. ONNX shape inference independently checks that shape below.
        source = runtime.get("x", initializers.get("x"))
        indices = runtime.get("indices", initializers.get("indices"))
        actual = [np.take(source, indices, axis=attrs["axis"])]
    elif op == "Gelu" and constant:
        # ReferenceEvaluator's context-dependent Gelu loader requires at least
        # one graph input to discover tensor types. Declaring each initializer
        # as an overridable input supplies types without changing the values.
        # Keep the actual conformance model constant-only for folding coverage.
        oracle = onnx.ModelProto.FromString(model.SerializeToString())
        oracle.graph.input.extend(
            h.make_tensor_value_info(
                name, nh.from_array(value).data_type, list(value.shape)
            )
            for name, value in initializers.items()
        )
        onnx.checker.check_model(
            onnx.shape_inference.infer_shapes(oracle), full_check=True
        )
        actual = ReferenceEvaluator(oracle).run(None, runtime)
    else:
        actual = ReferenceEvaluator(model).run(None, runtime)
    del model.graph.output[:]
    model.graph.output.extend(
        h.make_tensor_value_info(
            name, nh.from_array(value).data_type, list(value.shape)
        )
        for name, value in zip(outputs, actual)
    )
    onnx.checker.check_model(model)
    if op == "Gather":
        onnx.checker.check_model(model, full_check=True)
    suffix = f"_{variant}" if variant else ""
    if constant:
        suffix += "_constant"
    case = {
        "name": f"{op}_{version}{suffix}",
        "operator": op,
        "opset": version,
        "model": model.SerializeToString(deterministic=True).hex(),
        "inputs": {name: tensor(value) for name, value in runtime.items()},
        "outputs": [tensor(value) for value in actual],
    }
    if op == "Gather":
        case["required"] = True
    if op == "Cast" and variant:
        case["required"] = True
    if op == "Gelu" and variant == "exact":
        case["required"] = True
    if op == "AveragePool" and version >= 19:
        case["required"] = True
    if op == "LayerNormalization" and variant:
        case["required_cuda"] = True
        if variant.startswith("f32"):
            case["required"] = True
    if op in {"Cast", "Gather"}:
        case["exact_storage"] = True
    if op == "Clip" and variant == "edges":
        case["check_zero_sign"] = True
    if op == "Gather" and variant:
        case["check_zero_sign"] = True
    return case


def generate():
    assert onnx.__version__ == "1.21.0"
    cases = []
    for op, (minimum, maximum) in sorted(RANGES.items()):
        versions = {minimum, maximum}
        versions.update(
            s.since_version
            for s in onnx.defs.get_all_schemas_with_history()
            if s.name == op and s.domain == "" and minimum <= s.since_version <= maximum
        )
        for version in sorted(versions):
            cases.append(fixture(op, version))
    for version in [13, 18]:
        for constant in [False, True]:
            for op in ["Relu", "Clip"]:
                cases.append(fixture(op, version, "edges", constant))
            for dtype in ["int8", "uint8"]:
                for zero in [0, 3]:
                    cases.append(
                        fixture(
                            "QuantizeLinear",
                            version,
                            f"halfway_{dtype}_{zero}",
                            constant,
                        )
                    )
    for constant in [False, True]:
        cases.append(fixture("Gelu", 20, "exact", constant))
        for version in [19, 20]:
            cases.append(fixture("AveragePool", version, "dilation_one", constant))
        for variant in [
            "f32",
            "f32_transposed",
            "f16",
            "f16_transposed",
            "bf16",
            "bf16_transposed",
        ]:
            cases.append(fixture("LayerNormalization", 17, variant, constant))
        for version in [18, 20]:
            for variant in ["f16_f32", "bf16_f32", "f32_f16", "f32_bf16"]:
                cases.append(fixture("Cast", version, variant, constant))
        for variant in [
            "matrix_float32_int64",
            "matrix_float16_int32",
            "matrix_bfloat16_int64",
            "scalar_float32_int32",
            "scalar_float16_int64",
            "empty_float32_int64",
            "empty_float16_int32",
            "bits_int64_int64",
            "bits_bool_int32",
            "bits_float64_int64",
            "zero_float32_int64",
            "last_float32_int64",
        ]:
            cases.append(fixture("Gather", 13, variant, constant))
    return "".join(
        json.dumps(case, sort_keys=True, separators=(",", ":")) + "\n" for case in cases
    )


def tiny(output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "conformance.jsonl").write_text(generate())
