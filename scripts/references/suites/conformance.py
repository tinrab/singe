"""Regenerate the PyTorch conformance fixture: just references conformance.

Every case names a registry operation, its attributes, its inputs, and what PyTorch returns for them, or the error PyTorch raises.
Inputs are stored once each, as bytes, and a case may read one through a view (a transpose, a narrowing, or an expansion), which the Rust side applies in the same order.
singe/tests/kernels/conformance.rs replays every case on CPU eager, CUDA eager, and a compiled CUDA graph.

The samples follow PyTorch's OpInfo sample inputs: scalars, empty tensors, odd sizes, broadcasts, strided views, and special values for each data type.
PyTorch computes every reference on the CPU.
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "singe/tests/references/conformance"

DTYPES = {
    "Bool": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "U16": torch.uint16,
    "I16": torch.int16,
    "U32": torch.uint32,
    "I32": torch.int32,
    "U64": torch.uint64,
    "I64": torch.int64,
    "F16": torch.float16,
    "Bf16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
    "F8E4M3": torch.float8_e4m3fn,
    "F8E5M2": torch.float8_e5m2,
}
NAMES = {dtype: name for name, dtype in DTYPES.items()}
FLOATS = ["F16", "Bf16", "F32", "F64"]
BYTE_FLOATS = ["F8E4M3", "F8E5M2"]
SIGNED = ["I8", "I16", "I32", "I64"]
UNSIGNED = ["U8", "U16", "U32", "U64"]
ALL = list(DTYPES)


def is_float(dtype):
    return dtype in FLOATS or dtype in BYTE_FLOATS


# Value ranges a sample's inputs may ask for, so that domain-restricted operations see mostly valid inputs.
DOMAINS = {
    "any": (-4.0, 4.0),
    "unit": (-0.95, 0.95),
    "positive": (0.05, 4.0),
    "above_one": (1.05, 4.0),
}


class Pool:
    """Stores each input tensor once and names it by data type, shape, seed, and domain."""

    def __init__(self):
        self.tensors = {}

    def tensor(self, dtype, shape, seed=1, domain="any"):
        key = f"{dtype} {'x'.join(map(str, shape)) or 'scalar'} #{seed} {domain}"
        if key not in self.tensors:
            self.tensors[key] = values(dtype, shape, seed, domain)
        return key

    def special(self, dtype, count=12, seed=0):
        """Stores the special values of `dtype`: NaN, infinities, and signed zeros for floats, and extremes for integers."""
        key = f"{dtype} special{count} #{seed}"
        if key not in self.tensors:
            self.tensors[key] = special(dtype, count, seed)
        return key

    def explicit(self, name, tensor):
        key = f"{NAMES[tensor.dtype]} {name}"
        self.tensors.setdefault(key, tensor)
        return key


def values(dtype, shape, seed, domain):
    """Returns deterministic values of `shape`, distinct within the tensor where the data type allows it."""
    count = math.prod(shape)
    steps = [(index * 37 + seed * 11) % 97 for index in range(count)]
    if dtype == "Bool":
        data = torch.tensor([step % 3 == 0 for step in steps], dtype=torch.bool)
    elif dtype in SIGNED:
        low = 1 if domain in ("positive", "above_one") else -9
        data = torch.tensor(
            [low + step % (10 - low) for step in steps], dtype=torch.int64
        )
        data = data.to(DTYPES[dtype])
    elif dtype in UNSIGNED:
        low = 1 if domain in ("positive", "above_one") else 0
        data = torch.tensor([low + step % 19 for step in steps], dtype=torch.int64).to(
            DTYPES[dtype]
        )
    else:
        low, high = DOMAINS[domain]
        data = torch.tensor(
            [low + step / 96.0 * (high - low) for step in steps], dtype=torch.float64
        )
        data = data.to(DTYPES[dtype])
    return data.reshape(shape)


def special(dtype, count, seed):
    if dtype == "Bool":
        data = [True, False] * (count // 2)
    elif is_float(dtype):
        data = [
            math.nan,
            math.inf,
            -math.inf,
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -2.5,
            3.0,
            1e-3,
            -7.25,
        ]
    elif dtype in SIGNED:
        info = torch.iinfo(DTYPES[dtype])
        data = [
            info.min,
            info.max,
            -1,
            0,
            1,
            info.min + 1,
            info.max - 1,
            2,
            -2,
            7,
            -7,
            100,
        ]
    else:
        info = torch.iinfo(DTYPES[dtype])
        data = [0, 1, info.max, info.max - 1, 2, 7, 100, 3, 5, 11, 13, 17]
    data = data[seed % len(data) :] + data[: seed % len(data)]
    data = (data * (count // len(data) + 1))[:count]
    if dtype in UNSIGNED or dtype in SIGNED:
        # uint64 extremes do not fit the int64 tensor torch.tensor builds first.
        return torch.tensor(
            [value if value < 2**63 else value - 2**64 for value in data],
            dtype=torch.int64,
        ).to(DTYPES[dtype])
    if dtype == "Bool":
        return torch.tensor(data, dtype=torch.bool)
    return torch.tensor(data, dtype=torch.float64).to(DTYPES[dtype])


@dataclass
class Input:
    """A pool tensor, read through views applied in order."""

    key: str
    views: list = field(default_factory=list)

    def encode(self):
        return (
            {"tensor": self.key, "views": self.views}
            if self.views
            else {"tensor": self.key}
        )

    def build(self, pool):
        tensor = pool.tensors[self.key]
        for view in self.views:
            match view:
                case ["transpose", first, second]:
                    tensor = tensor.transpose(first, second)
                case ["narrow", dim, start, length]:
                    tensor = tensor.narrow(dim, start, length)
                case ["expand", shape]:
                    tensor = tensor.expand(shape)
        return tensor


@dataclass
class Sample:
    """One call of an operation: a label, its inputs, its attributes in the Rust test's encoding, and the PyTorch reference."""

    label: str
    inputs: list
    reference: object
    attributes: object = None
    # Whether the operation is a tensor method Singe composes from registry operations, called with `attributes` as its arguments.
    is_composite: bool = False

    def composite(self):
        self.is_composite = True
        return self


def plain(pool, dtype, shape, seed=1, domain="any"):
    return Input(pool.tensor(dtype, shape, seed, domain))


def transposed(pool, dtype, shape, seed=1, domain="any"):
    """Returns a transposed view of `shape`, read from a contiguous tensor with its first two axes swapped."""
    base = [shape[1], shape[0], *shape[2:]]
    return Input(pool.tensor(dtype, base, seed, domain), [["transpose", 0, 1]])


def sliced(pool, dtype, shape, seed=1, domain="any"):
    """Returns a view of `shape` with a storage offset and row padding, narrowed from a larger tensor."""
    base = [shape[0] + 2, shape[1] + 3, *shape[2:]]
    return Input(
        pool.tensor(dtype, base, seed, domain),
        [["narrow", 0, 1, shape[0]], ["narrow", 1, 2, shape[1]]],
    )


def expanded(pool, dtype, shape, seed=1, domain="any"):
    """Returns `shape` expanded from one row, so its first axis has stride zero."""
    return Input(pool.tensor(dtype, [1, *shape[1:]], seed, domain), [["expand", shape]])


def unary_samples(pool, dtype, reference, domain="any"):
    samples = [
        Sample("scalar", [plain(pool, dtype, [], 1, domain)], reference),
        Sample("empty", [plain(pool, dtype, [0, 3], 1, domain)], reference),
        Sample("odd", [plain(pool, dtype, [7], 1, domain)], reference),
        Sample("matrix", [plain(pool, dtype, [3, 4], 1, domain)], reference),
        Sample("transposed", [transposed(pool, dtype, [3, 4], 1, domain)], reference),
        Sample("sliced", [sliced(pool, dtype, [3, 4], 1, domain)], reference),
        Sample("expanded", [expanded(pool, dtype, [3, 4], 1, domain)], reference),
    ]
    if dtype != "Bool":
        samples.append(Sample("special", [Input(pool.special(dtype))], reference))
    return samples


def binary_samples(pool, dtype, reference, domains=("any", "any")):
    left, right = domains

    def pair(first, second):
        return [first, second]

    samples = [
        Sample(
            "scalar",
            pair(plain(pool, dtype, [], 1, left), plain(pool, dtype, [], 2, right)),
            reference,
        ),
        Sample(
            "empty",
            pair(
                plain(pool, dtype, [0, 3], 1, left),
                plain(pool, dtype, [0, 3], 2, right),
            ),
            reference,
        ),
        Sample(
            "matrix",
            pair(
                plain(pool, dtype, [3, 4], 1, left),
                plain(pool, dtype, [3, 4], 2, right),
            ),
            reference,
        ),
        Sample(
            "row broadcast",
            pair(
                plain(pool, dtype, [3, 4], 1, left), plain(pool, dtype, [4], 2, right)
            ),
            reference,
        ),
        Sample(
            "outer broadcast",
            pair(
                plain(pool, dtype, [3, 1], 1, left),
                plain(pool, dtype, [1, 4], 2, right),
            ),
            reference,
        ),
        Sample(
            "scalar operand",
            pair(plain(pool, dtype, [3, 4], 1, left), plain(pool, dtype, [], 2, right)),
            reference,
        ),
        Sample(
            "transposed",
            pair(
                transposed(pool, dtype, [3, 4], 1, left),
                sliced(pool, dtype, [3, 4], 2, right),
            ),
            reference,
        ),
    ]
    if dtype != "Bool":
        samples.append(
            Sample(
                "special",
                pair(Input(pool.special(dtype)), Input(pool.special(dtype, 12, 5))),
                reference,
            )
        )
    return samples


UNARY = {
    "abs": (torch.abs, "any"),
    "acos": (torch.acos, "unit"),
    "acosh": (torch.acosh, "above_one"),
    "asin": (torch.asin, "unit"),
    "asinh": (torch.asinh, "any"),
    "atan": (torch.atan, "any"),
    "bitwise_not": (torch.bitwise_not, "any"),
    "atanh": (torch.atanh, "unit"),
    "ceil": (torch.ceil, "any"),
    "contiguous": (lambda x: x.contiguous(), "any"),
    "copy": (torch.clone, "any"),
    "cos": (torch.cos, "any"),
    "cosh": (torch.cosh, "any"),
    "erf": (torch.erf, "any"),
    "log1p": (torch.log1p, "any"),
    "expm1": (torch.expm1, "any"),
    "round": (torch.round, "any"),
    "isnan": (torch.isnan, "any"),
    "exp": (torch.exp, "any"),
    "floor": (torch.floor, "any"),
    "gelu": (lambda x: functional.gelu(x, approximate="tanh"), "any"),
    "gelu_exact": (lambda x: functional.gelu(x, approximate="none"), "any"),
    "log": (torch.log, "positive"),
    "logical_not": (torch.logical_not, "any"),
    "neg": (torch.neg, "any"),
    "reciprocal": (torch.reciprocal, "positive"),
    "relu": (torch.relu, "any"),
    "rsqrt": (torch.rsqrt, "positive"),
    "sigmoid": (torch.sigmoid, "any"),
    "silu": (functional.silu, "any"),
    "sin": (torch.sin, "any"),
    "sinh": (torch.sinh, "any"),
    "sqrt": (torch.sqrt, "positive"),
    "square": (torch.square, "any"),
    "tan": (torch.tan, "unit"),
    "tanh": (torch.tanh, "any"),
    "trunc": (torch.trunc, "any"),
}

BINARY = {
    "add": (torch.add, ("any", "any")),
    "atan2": (torch.atan2, ("any", "any")),
    "bitwise_and": (torch.bitwise_and, ("any", "any")),
    "bitwise_left_shift": (torch.bitwise_left_shift, ("any", "positive")),
    "bitwise_or": (torch.bitwise_or, ("any", "any")),
    "bitwise_right_shift": (torch.bitwise_right_shift, ("any", "positive")),
    "bitwise_xor": (torch.bitwise_xor, ("any", "any")),
    "div": (torch.div, ("any", "positive")),
    "equal": (torch.eq, ("any", "any")),
    "greater": (torch.gt, ("any", "any")),
    "greater_equal": (torch.ge, ("any", "any")),
    "less": (torch.lt, ("any", "any")),
    "less_equal": (torch.le, ("any", "any")),
    "logical_and": (torch.logical_and, ("any", "any")),
    "logical_or": (torch.logical_or, ("any", "any")),
    "maximum": (torch.maximum, ("any", "any")),
    "minimum": (torch.minimum, ("any", "any")),
    "modulo": (torch.remainder, ("any", "positive")),
    "mul": (torch.mul, ("any", "any")),
    "not_equal": (torch.ne, ("any", "any")),
    "pow": (torch.pow, ("positive", "any")),
    "remainder": (torch.remainder, ("any", "positive")),
    "sub": (torch.sub, ("any", "any")),
    "swiglu": (lambda x, gate: x * functional.silu(gate), ("any", "any")),
}

SCALAR = {
    "add_scalar": lambda x, s: x + s,
    "sub_scalar": lambda x, s: x - s,
    "rsub_scalar": lambda x, s: s - x,
    "scale": lambda x, s: x * s,
    "rdiv_scalar": lambda x, s: s / x,
    "pow_scalar": lambda x, s: x**s,
}


def scalar_samples(pool, dtype, function):
    samples = []
    for kind, value in [("float", 0.75), ("int", 3), ("bool", True)]:
        attributes = {"Scalar": {kind: value}}

        def reference(x, value=value):
            return function(x, value)

        for label, operand in [
            ("matrix", plain(pool, dtype, [3, 4], 1, "positive")),
            ("scalar", plain(pool, dtype, [], 1, "positive")),
            ("empty", plain(pool, dtype, [0, 3])),
            ("transposed", transposed(pool, dtype, [3, 4], 1, "positive")),
        ]:
            samples.append(Sample(f"{kind} {label}", [operand], reference, attributes))
        if dtype != "Bool":
            samples.append(
                Sample(
                    f"{kind} special",
                    [Input(pool.special(dtype))],
                    reference,
                    attributes,
                )
            )
    return samples


def reduce(function, dims, keepdim):
    """Returns a reference that reduces over `dims`, or every axis when `dims` is empty, as Singe's reductions do."""

    def reference(x):
        axes = sorted(dim % x.dim() for dim in dims) if dims else list(range(x.dim()))
        if function is torch.prod and x.dim() == 0:
            return torch.prod(x)
        if function is torch.prod:
            result = x
            for axis in reversed(axes):
                result = torch.prod(result, dim=axis, keepdim=True)
            if not keepdim:
                result = result.reshape(
                    [size for axis, size in enumerate(x.shape) if axis not in axes]
                )
            return result
        return function(x, dim=axes, keepdim=keepdim)

    return reference


REDUCTIONS = {
    "sum": torch.sum,
    "mean": torch.mean,
    "max": torch.amax,
    "min": torch.amin,
    "product": torch.prod,
}


def reduction_samples(pool, dtype, function):
    samples = []
    for label, operand, dims, keepdim in [
        ("matrix rows", plain(pool, dtype, [3, 4]), [1], False),
        ("matrix columns keepdim", plain(pool, dtype, [3, 4]), [0], True),
        ("matrix all", plain(pool, dtype, [3, 4]), [], False),
        ("matrix both keepdim", plain(pool, dtype, [3, 4]), [0, 1], True),
        ("rank3 outer", plain(pool, dtype, [2, 3, 4]), [0, -1], False),
        ("scalar", plain(pool, dtype, []), [], False),
        ("empty rows", plain(pool, dtype, [0, 4]), [1], False),
        ("empty columns", plain(pool, dtype, [0, 4]), [0], False),
        ("transposed", transposed(pool, dtype, [3, 4]), [1], False),
        ("expanded", expanded(pool, dtype, [3, 4]), [0], False),
    ]:
        attributes = {"Reduction": {"dimensions": dims, "keepdim": keepdim}}
        samples.append(
            Sample(label, [operand], reduce(function, dims, keepdim), attributes)
        )
    if dtype != "Bool":
        samples.append(
            Sample(
                "special",
                [Input(pool.special(dtype))],
                reduce(function, [], False),
                {"Reduction": {"dimensions": [], "keepdim": False}},
            )
        )
    return samples


DIMENSION = {
    "softmax": lambda x, dim: torch.softmax(x, dim),
    "log_softmax": lambda x, dim: torch.log_softmax(x, dim),
    "argmax": lambda x, dim: torch.argmax(x, dim),
    "argmin": lambda x, dim: torch.argmin(x, dim),
    "cumsum": lambda x, dim: torch.cumsum(x, dim),
    "cumprod": lambda x, dim: torch.cumprod(x, dim),
}


def dimension_samples(pool, dtype, function):
    samples = []
    for label, operand, dim in [
        ("matrix first", plain(pool, dtype, [3, 4]), 0),
        ("matrix last", plain(pool, dtype, [3, 4]), -1),
        ("odd", plain(pool, dtype, [7]), 0),
        ("scalar", plain(pool, dtype, []), 0),
        ("empty", plain(pool, dtype, [0, 4]), 1),
        ("transposed", transposed(pool, dtype, [3, 4]), 1),
    ]:
        samples.append(
            Sample(
                label,
                [operand],
                lambda x, dim=dim: function(x, dim),
                {"Dimension": dim},
            )
        )
    if dtype != "Bool":
        samples.append(
            Sample(
                "special",
                [Input(pool.special(dtype))],
                lambda x: function(x, 0),
                {"Dimension": 0},
            )
        )
    return samples


def sort_samples(pool, dtype):
    samples = []
    for label, operand, dim, descending in [
        ("rows", plain(pool, dtype, [3, 5]), 1, False),
        ("rows descending", plain(pool, dtype, [3, 5]), 1, True),
        ("columns", plain(pool, dtype, [3, 5]), 0, False),
        ("transposed", transposed(pool, dtype, [3, 5]), -1, True),
        ("empty", plain(pool, dtype, [0, 4]), 1, False),
    ]:

        def reference(x, dim=dim, descending=descending):
            return torch.argsort(x, dim=dim, descending=descending, stable=True)

        samples.append(
            Sample(
                label,
                [operand],
                reference,
                {"Sort": {"dim": dim, "descending": descending}},
            )
        )
    if dtype != "Bool":

        def special_reference(x):
            return torch.argsort(x, dim=0, stable=True)

        samples.append(
            Sample(
                "special",
                [Input(pool.special(dtype))],
                special_reference,
                {"Sort": {"dim": 0, "descending": False}},
            )
        )
    return samples


def topk_samples(pool, dtype, outputs):
    samples = []
    for label, operand, k, dim, largest in [
        ("rows largest", plain(pool, dtype, [3, 5]), 2, 1, True),
        ("rows smallest", plain(pool, dtype, [3, 5]), 3, -1, False),
        ("columns", plain(pool, dtype, [3, 5]), 1, 0, True),
        ("transposed", transposed(pool, dtype, [3, 5]), 2, 1, True),
        ("zero k", plain(pool, dtype, [3, 5]), 0, 1, True),
    ]:

        def reference(x, k=k, dim=dim, largest=largest):
            result = torch.topk(x, k, dim=dim, largest=largest, sorted=True)
            return outputs(result)

        attributes = {"TopK": {"k": k, "dim": dim, "largest": largest, "sorted": True}}
        samples.append(Sample(label, [operand], reference, attributes))
    return samples


def clamp_samples(pool, dtype):
    attributes = {"Clamp": {"min": -1.25, "max": 2.5}}

    def reference(x):
        return torch.clamp(x, -1.25, 2.5)

    samples = [
        Sample("matrix", [plain(pool, dtype, [3, 4])], reference, attributes),
        Sample("scalar", [plain(pool, dtype, [])], reference, attributes),
        Sample("transposed", [transposed(pool, dtype, [3, 4])], reference, attributes),
    ]
    if dtype != "Bool":
        samples.append(
            Sample("special", [Input(pool.special(dtype))], reference, attributes)
        )
    return samples


def lerp_samples(pool, dtype):
    attributes = {"Lerp": 0.25}

    def reference(start, end):
        return torch.lerp(start, end, 0.25)

    return [
        Sample(
            "matrix",
            [plain(pool, dtype, [3, 4]), plain(pool, dtype, [3, 4], 2)],
            reference,
            attributes,
        ),
        Sample(
            "broadcast",
            [plain(pool, dtype, [3, 4]), plain(pool, dtype, [4], 2)],
            reference,
            attributes,
        ),
        Sample(
            "transposed",
            [transposed(pool, dtype, [3, 4]), plain(pool, dtype, [3, 4], 2)],
            reference,
            attributes,
        ),
        Sample(
            "special",
            [Input(pool.special(dtype)), Input(pool.special(dtype, 12, 5))],
            reference,
            attributes,
        )
        if dtype != "Bool"
        else None,
    ]


def where_samples(pool, dtype):
    return [
        Sample(
            "matrix",
            [
                plain(pool, "Bool", [3, 4], 3),
                plain(pool, dtype, [3, 4]),
                plain(pool, dtype, [3, 4], 2),
            ],
            torch.where,
        ),
        Sample(
            "broadcast",
            [
                plain(pool, "Bool", [3, 1], 3),
                plain(pool, dtype, [4]),
                plain(pool, dtype, [], 2),
            ],
            torch.where,
        ),
        Sample(
            "transposed",
            [
                plain(pool, "Bool", [3, 4], 3),
                transposed(pool, dtype, [3, 4]),
                sliced(pool, dtype, [3, 4], 2),
            ],
            torch.where,
        ),
        Sample(
            "empty",
            [
                plain(pool, "Bool", [0, 4], 3),
                plain(pool, dtype, [0, 4]),
                plain(pool, dtype, [0, 4], 2),
            ],
            torch.where,
        ),
    ]


def shape_samples(pool, dtype):
    """Returns samples of the shape and view operations, which every data type supports."""
    matrix = plain(pool, dtype, [3, 4])
    rank3 = plain(pool, dtype, [2, 3, 4])
    flat = plain(pool, dtype, [12])

    def transpose(dim0, dim1):
        return {"Transpose": {"dim0": dim0, "dim1": dim1}}

    def slice_attributes(dim, start, length, step):
        return {"Slice": {"dim": dim, "start": start, "length": length, "step": step}}

    def slice_reference(dim, start, length, step):
        def reference(x):
            index = [slice(None)] * x.dim()
            index[dim] = slice(
                start, start + (length - 1) * step + 1 if length else start, step
            )
            return x[tuple(index)]

        return reference

    def strided(dimensions, strides, offset):
        return {
            "Strided": {"dimensions": dimensions, "strides": strides, "offset": offset}
        }

    def flatten(start, end):
        return {"Flatten": {"start_dim": start, "end_dim": end}}

    return {
        "transpose": [
            Sample("matrix", [matrix], lambda x: x.transpose(0, 1), transpose(0, 1)),
            Sample(
                "rank3 outer", [rank3], lambda x: x.transpose(0, 2), transpose(0, 2)
            ),
            Sample(
                "negative", [rank3], lambda x: x.transpose(-1, -2), transpose(-1, -2)
            ),
        ],
        "permute": [
            Sample(
                "rank3", [rank3], lambda x: x.permute(2, 0, 1), {"Permute": [2, 0, 1]}
            ),
            Sample(
                "transposed",
                [transposed(pool, dtype, [3, 4])],
                lambda x: x.permute(1, 0),
                {"Permute": [1, 0]},
            ),
        ],
        "reshape": [
            Sample("matrix", [matrix], lambda x: x.reshape(2, 6), {"Shape": [2, 6]}),
            Sample("flatten", [rank3], lambda x: x.reshape(24), {"Shape": [24]}),
            Sample(
                "transposed",
                [transposed(pool, dtype, [3, 4])],
                lambda x: x.reshape(12),
                {"Shape": [12]},
            ),
            Sample(
                "scalar",
                [plain(pool, dtype, [])],
                lambda x: x.reshape(1, 1),
                {"Shape": [1, 1]},
            ),
            Sample(
                "empty",
                [plain(pool, dtype, [0, 4])],
                lambda x: x.reshape(4, 0),
                {"Shape": [4, 0]},
            ),
        ],
        "squeeze": [
            Sample(
                "middle",
                [plain(pool, dtype, [3, 1, 4])],
                lambda x: x.squeeze(1),
                {"Dimension": 1},
            ),
            Sample(
                "negative",
                [plain(pool, dtype, [3, 1, 4])],
                lambda x: x.squeeze(-2),
                {"Dimension": -2},
            ),
            Sample("not one", [matrix], lambda x: x.squeeze(0), {"Dimension": 0}),
        ],
        "unsqueeze": [
            Sample("front", [matrix], lambda x: x.unsqueeze(0), {"Dimension": 0}),
            Sample("back", [matrix], lambda x: x.unsqueeze(2), {"Dimension": 2}),
            Sample("negative", [matrix], lambda x: x.unsqueeze(-1), {"Dimension": -1}),
            Sample(
                "scalar",
                [plain(pool, dtype, [])],
                lambda x: x.unsqueeze(0),
                {"Dimension": 0},
            ),
        ],
        "slice": [
            Sample(
                "rows",
                [matrix],
                slice_reference(0, 1, 2, 1),
                slice_attributes(0, 1, 2, 1),
            ),
            Sample(
                "stepped",
                [matrix],
                slice_reference(1, 0, 2, 2),
                slice_attributes(1, 0, 2, 2),
            ),
            Sample(
                "empty",
                [matrix],
                slice_reference(1, 2, 0, 1),
                slice_attributes(1, 2, 0, 1),
            ),
            Sample(
                "negative dim",
                [rank3],
                slice_reference(2, 1, 3, 1),
                slice_attributes(-1, 1, 3, 1),
            ),
        ],
        "concat": [
            Sample(
                "rows",
                [matrix, plain(pool, dtype, [3, 4], 2)],
                lambda x, y: torch.cat([x, y], 0),
                {"Concat": 0},
            ),
            Sample(
                "columns",
                [matrix, plain(pool, dtype, [3, 2], 2)],
                lambda x, y: torch.cat([x, y], 1),
                {"Concat": 1},
            ),
            Sample(
                "three",
                [
                    plain(pool, dtype, [2]),
                    plain(pool, dtype, [3], 2),
                    plain(pool, dtype, [1], 3),
                ],
                lambda *xs: torch.cat(xs, 0),
                {"Concat": 0},
            ),
            Sample(
                "with empty",
                [matrix, plain(pool, dtype, [0, 4], 2)],
                lambda x, y: torch.cat([x, y], 0),
                {"Concat": 0},
            ),
            Sample(
                "transposed",
                [transposed(pool, dtype, [3, 4]), matrix],
                lambda x, y: torch.cat([x, y], -1),
                {"Concat": -1},
            ),
        ],
        "flatten": [
            Sample("leading", [rank3], lambda x: x.flatten(0, 1), flatten(0, 1)),
            Sample("trailing", [rank3], lambda x: x.flatten(1, -1), flatten(1, -1)),
            Sample("all", [rank3], lambda x: x.flatten(0, -1), flatten(0, -1)),
        ],
        "expand": [
            Sample(
                "column",
                [plain(pool, dtype, [3, 1])],
                lambda x: x.expand(3, 4),
                {"Shape": [3, 4]},
            ),
            Sample(
                "new axes",
                [plain(pool, dtype, [4])],
                lambda x: x.expand(2, 3, 4),
                {"Shape": [2, 3, 4]},
            ),
            Sample(
                "scalar",
                [plain(pool, dtype, [])],
                lambda x: x.expand(2, 2),
                {"Shape": [2, 2]},
            ),
        ],
        "as_strided": [
            Sample(
                "offset",
                [flat],
                lambda x: x.as_strided([3, 3], [3, 1], 1),
                strided([3, 3], [3, 1], 1),
            ),
            Sample(
                "overlapping",
                [flat],
                lambda x: x.as_strided([3, 3], [1, 1], 0),
                strided([3, 3], [1, 1], 0),
            ),
            Sample(
                "broadcast",
                [flat],
                lambda x: x.as_strided([2, 4], [0, 2], 2),
                strided([2, 4], [0, 2], 2),
            ),
        ],
        "strided_scatter": [
            Sample(
                "offset",
                [flat, plain(pool, dtype, [2, 3], 2)],
                lambda x, src: torch.as_strided_scatter(x, src, [2, 3], [4, 1], 1),
                strided([2, 3], [4, 1], 1),
            ),
        ],
        "repeat_kv": [
            Sample(
                "heads",
                [plain(pool, dtype, [1, 2, 3, 4])],
                lambda x: x.repeat_interleave(2, dim=1),
                {"RepeatKv": 2},
            ),
            Sample(
                "one",
                [plain(pool, dtype, [1, 2, 3, 4])],
                lambda x: x.clone(),
                {"RepeatKv": 1},
            ),
        ],
        "pad": [
            Sample(
                "matrix",
                [matrix],
                lambda x: functional.pad(x, (2, 1, 1, 0), value=0.5),
                {"Pad": {"padding": [[1, 0], [2, 1]], "value": 0.5}},
            ),
            Sample(
                "crop",
                [matrix],
                lambda x: functional.pad(x, (-1, 1, 0, 0), value=0.0),
                {"Pad": {"padding": [[0, 0], [-1, 1]], "value": 0.0}},
            ),
        ],
    }


def put_columns(x, values, indices):
    result = x.clone()
    result[:, indices] = values
    return result


def indexing_samples(pool, dtype):
    table = plain(pool, dtype, [4, 3])
    indices = Input(pool.explicit("indices [2,0,3]", torch.tensor([2, 0, 3])))
    repeated = Input(pool.explicit("indices [1,1]", torch.tensor([1, 1])))
    negative = Input(pool.explicit("indices [-1,0]", torch.tensor([-1, 0])))
    out_of_range = Input(pool.explicit("indices [4]", torch.tensor([4])))
    gather_indices = Input(
        pool.explicit("indices 4x2", torch.tensor([[0, 2], [1, 1], [2, 0], [0, 0]]))
    )
    lookup = Input(pool.explicit("indices 2x3", torch.tensor([[0, 3, 1], [2, 2, 0]])))
    return {
        "index_select": [
            Sample(
                "rows",
                [table, indices],
                lambda x, i: torch.index_select(x, 0, i),
                {"IndexSelect": 0},
            ),
            Sample(
                "repeated columns",
                [table, repeated],
                lambda x, i: torch.index_select(x, 1, i),
                {"IndexSelect": 1},
            ),
            Sample(
                "transposed",
                [transposed(pool, dtype, [4, 3]), indices],
                lambda x, i: torch.index_select(x, 0, i),
                {"IndexSelect": 0},
            ),
            Sample(
                "out of range",
                [table, out_of_range],
                lambda x, i: torch.index_select(x, 0, i),
                {"IndexSelect": 0},
            ),
        ],
        "gather": [
            Sample(
                "rows",
                [table, gather_indices],
                lambda x, i: torch.gather(x, 1, i),
                {"dim": 1},
            ).composite(),
            Sample(
                "columns",
                [
                    table,
                    Input(
                        pool.explicit(
                            "indices 2x3", torch.tensor([[0, 3, 1], [2, 2, 0]])
                        )
                    ),
                ],
                lambda x, i: torch.gather(x, 0, i),
                {"dim": 0},
            ).composite(),
            Sample(
                "smaller",
                [table, Input(pool.explicit("indices 2x1", torch.tensor([[2], [0]])))],
                lambda x, i: torch.gather(x, 1, i),
                {"dim": 1},
            ).composite(),
        ],
        "take": [
            Sample("rows", [table, indices], lambda x, i: x[i], {"IndexSelect": 0}),
            Sample(
                "matrix of columns",
                [table, gather_indices],
                lambda x, i: x[:, i],
                {"IndexSelect": 1},
            ),
            Sample(
                "negative", [table, negative], lambda x, i: x[i], {"IndexSelect": 0}
            ),
            Sample(
                "scalar index",
                [table, Input(pool.explicit("index 2", torch.tensor(2)))],
                lambda x, i: x[:, i],
                {"IndexSelect": 1},
            ),
        ],
        "embedding": [
            Sample(
                "lookup",
                [table, lookup],
                lambda weight, i: functional.embedding(i, weight),
            ),
            Sample(
                "out of range",
                [table, out_of_range],
                lambda weight, i: functional.embedding(i, weight),
            ),
        ],
        "index": [
            Sample("rows", [table, indices], lambda x, i: x[i], {"Dimension": 0}),
            Sample("negative", [table, negative], lambda x, i: x[i], {"Dimension": 0}),
            Sample(
                "columns", [table, repeated], lambda x, i: x[:, i], {"Dimension": 1}
            ),
            Sample(
                "pairs",
                [
                    plain(pool, dtype, [2, 4, 3]),
                    indices,
                    Input(pool.explicit("indices [0,2,1]", torch.tensor([0, 2, 1]))),
                ],
                lambda x, i, j: x[:, i, j],
                {"Dimension": 1},
            ),
            Sample(
                "out of range",
                [table, out_of_range],
                lambda x, i: x[i],
                {"Dimension": 0},
            ),
        ],
        "index_put": [
            Sample(
                "rows",
                [table, plain(pool, dtype, [3, 3], 2), indices],
                lambda x, v, i: x.index_put((i,), v),
                {"IndexPut": {"dim": 0, "accumulate": False}},
            ),
            Sample(
                "accumulate",
                [table, plain(pool, dtype, [2, 3], 2), repeated],
                lambda x, v, i: x.index_put((i,), v, accumulate=True),
                {"IndexPut": {"dim": 0, "accumulate": True}},
            ),
            Sample(
                "columns",
                [table, plain(pool, dtype, [4, 2], 2), negative],
                put_columns,
                {"IndexPut": {"dim": 1, "accumulate": False}},
            ),
        ],
    }


def matmul_samples(pool, dtype):
    def pair(lhs, rhs, seed_lhs=1, seed_rhs=2):
        return [plain(pool, dtype, lhs, seed_lhs), plain(pool, dtype, rhs, seed_rhs)]

    return [
        Sample("matrix", pair([3, 4], [4, 5]), torch.matmul),
        Sample("batched lhs", pair([2, 3, 4], [4, 5]), torch.matmul),
        Sample("batched", pair([2, 3, 4], [2, 4, 5]), torch.matmul),
        Sample("broadcast batch", pair([2, 1, 3, 4], [3, 4, 2]), torch.matmul),
        Sample("vector lhs", pair([4], [4, 5]), torch.matmul),
        Sample("vector rhs", pair([3, 4], [4]), torch.matmul),
        Sample("dot", pair([4], [4]), torch.matmul),
        Sample(
            "transposed",
            [transposed(pool, dtype, [3, 4]), transposed(pool, dtype, [4, 5], 2)],
            torch.matmul,
        ),
        Sample("empty rows", pair([0, 4], [4, 5]), torch.matmul),
        Sample("empty inner", pair([3, 0], [0, 5]), torch.matmul),
    ]


def contraction_samples(pool, dtype):
    def reference(alpha, beta, equation):
        def contract(a, b, c):
            return alpha * torch.einsum(equation, a, b) + beta * c

        return contract

    return [
        Sample(
            "matrix",
            [
                plain(pool, dtype, [3, 4]),
                plain(pool, dtype, [4, 5], 2),
                plain(pool, dtype, [3, 5], 3),
            ],
            reference(1.5, 0.5, "ik,kj->ij"),
            {
                "Contraction": {
                    "a_mode": [0, 1],
                    "b_mode": [1, 2],
                    "c_mode": [0, 2],
                    "d_mode": [0, 2],
                    "alpha": 1.5,
                    "beta": 0.5,
                }
            },
        ),
        Sample(
            "batched",
            [
                plain(pool, dtype, [2, 3, 4]),
                plain(pool, dtype, [2, 4, 5], 2),
                plain(pool, dtype, [2, 3, 5], 3),
            ],
            reference(1.0, 0.0, "bik,bkj->bij"),
            {
                "Contraction": {
                    "a_mode": [0, 1, 2],
                    "b_mode": [0, 2, 3],
                    "c_mode": [0, 1, 3],
                    "d_mode": [0, 1, 3],
                    "alpha": 1.0,
                    "beta": 0.0,
                }
            },
        ),
    ]


def normalization_samples(pool, dtype):
    def layer(normalized, weight, bias):
        return {
            "LayerNormalization": {
                "normalized_shape": normalized,
                "has_weight": weight,
                "has_bias": bias,
                "eps": 1e-5,
            }
        }

    x = plain(pool, dtype, [3, 4])
    weight = plain(pool, dtype, [4], 2)
    bias = plain(pool, dtype, [4], 3)
    rank3 = plain(pool, dtype, [2, 3, 4])
    plane_weight = plain(pool, dtype, [3, 4], 2)
    return {
        "layer_norm": [
            Sample(
                "affine",
                [x, weight, bias],
                lambda x, w, b: functional.layer_norm(x, (4,), w, b, 1e-5),
                layer([4], True, True),
            ),
            Sample(
                "plain",
                [x],
                lambda x: functional.layer_norm(x, (4,), None, None, 1e-5),
                layer([4], False, False),
            ),
            Sample(
                "plane",
                [rank3, plane_weight],
                lambda x, w: functional.layer_norm(x, (3, 4), w, None, 1e-5),
                layer([3, 4], True, False),
            ),
            Sample(
                "transposed",
                [transposed(pool, dtype, [3, 4]), weight, bias],
                lambda x, w, b: functional.layer_norm(x, (4,), w, b, 1e-5),
                layer([4], True, True),
            ),
        ],
        "rms_norm": [
            Sample(
                "weight",
                [x, weight],
                lambda x, w: functional.rms_norm(x, (4,), w, 1e-5),
                layer([4], True, False),
            ),
            Sample(
                "plain",
                [x],
                lambda x: functional.rms_norm(x, (4,), None, 1e-5),
                layer([4], False, False),
            ),
            Sample(
                "plane",
                [rank3, plane_weight],
                lambda x, w: functional.rms_norm(x, (3, 4), w, 1e-5),
                layer([3, 4], True, False),
            ),
            Sample(
                "transposed",
                [transposed(pool, dtype, [3, 4]), weight],
                lambda x, w: functional.rms_norm(x, (4,), w, 1e-5),
                layer([4], True, False),
            ),
        ],
        "batch_norm": [
            Sample(
                "inference",
                [
                    plain(pool, dtype, [2, 3, 2, 2]),
                    plain(pool, dtype, [3], 2),
                    plain(pool, dtype, [3], 3),
                    plain(pool, dtype, [3], 4),
                    plain(pool, dtype, [3], 5, "positive"),
                ],
                lambda x, w, b, m, v: functional.batch_norm(
                    x, m, v, w, b, False, 0.1, 1e-5
                ),
                {
                    "BatchNormalization": {
                        "has_weight": True,
                        "has_bias": True,
                        "has_running_mean": True,
                        "has_running_var": True,
                        "training": False,
                        "momentum": 0.1,
                        "eps": 1e-5,
                    }
                },
            ),
            Sample(
                "batch statistics",
                [plain(pool, dtype, [2, 3, 2, 2])],
                lambda x: functional.batch_norm(
                    x, None, None, None, None, True, 0.1, 1e-5
                ),
                {
                    "BatchNormalization": {
                        "has_weight": False,
                        "has_bias": False,
                        "has_running_mean": False,
                        "has_running_var": False,
                        "training": True,
                        "momentum": 0.1,
                        "eps": 1e-5,
                    }
                },
            ),
        ],
    }


def convolution_samples(pool, dtype):
    def attributes(
        bias, stride=(1, 1), padding=(0, 0, 0, 0), dilation=(1, 1), groups=1
    ):
        return {
            "Conv2d": {
                "has_bias": bias,
                "stride": list(stride),
                "padding": list(padding),
                "dilation": list(dilation),
                "groups": groups,
            }
        }

    def conv(stride=1, padding=(0, 0, 0, 0), dilation=1, groups=1):
        top, bottom, left, right = padding

        def reference(x, w, b=None):
            x = functional.pad(x, (left, right, top, bottom))
            return functional.conv2d(x, w, b, stride, 0, dilation, groups)

        return reference

    x = plain(pool, dtype, [1, 2, 5, 5])
    weight = plain(pool, dtype, [3, 2, 3, 3], 2)
    bias = plain(pool, dtype, [3], 3)
    return [
        Sample("valid", [x, weight, bias], conv(), attributes(True)),
        Sample(
            "padded",
            [x, weight],
            conv(padding=(1, 1, 1, 1)),
            attributes(False, padding=(1, 1, 1, 1)),
        ),
        Sample(
            "asymmetric padding",
            [x, weight],
            conv(padding=(1, 0, 0, 2)),
            attributes(False, padding=(1, 0, 0, 2)),
        ),
        Sample(
            "strided",
            [x, weight, bias],
            conv(stride=2),
            attributes(True, stride=(2, 2)),
        ),
        Sample(
            "dilated", [x, weight], conv(dilation=2), attributes(False, dilation=(2, 2))
        ),
        Sample(
            "grouped",
            [plain(pool, dtype, [1, 4, 4, 4]), plain(pool, dtype, [2, 2, 3, 3], 2)],
            conv(groups=2),
            attributes(False, groups=2),
        ),
    ]


def pooling_samples(pool, dtype):
    def attributes(
        kernel,
        stride,
        padding=(0, 0, 0, 0),
        dilation=(1, 1),
        ceil=False,
        include_pad=None,
    ):
        return {
            "Pool2d": {
                "kernel_size": list(kernel),
                "stride": list(stride),
                "padding": list(padding),
                "dilation": list(dilation),
                "ceil_mode": ceil,
                "count_include_pad": include_pad,
            }
        }

    x = plain(pool, dtype, [1, 2, 5, 5])
    return {
        "max_pool2d": [
            Sample(
                "windows",
                [x],
                lambda x: functional.max_pool2d(x, 2, 2),
                attributes((2, 2), (2, 2)),
            ),
            Sample(
                "padded",
                [x],
                lambda x: functional.max_pool2d(x, 3, 1, 1),
                attributes((3, 3), (1, 1), (1, 1, 1, 1)),
            ),
            Sample(
                "ceil",
                [x],
                lambda x: functional.max_pool2d(x, 2, 2, ceil_mode=True),
                attributes((2, 2), (2, 2), ceil=True),
            ),
            Sample(
                "dilated",
                [x],
                lambda x: functional.max_pool2d(x, 2, 1, dilation=2),
                attributes((2, 2), (1, 1), dilation=(2, 2)),
            ),
        ],
        "avg_pool2d": [
            Sample(
                "windows",
                [x],
                lambda x: functional.avg_pool2d(x, 2, 2),
                attributes((2, 2), (2, 2), include_pad=True),
            ),
            Sample(
                "padded with pad",
                [x],
                lambda x: functional.avg_pool2d(x, 3, 1, 1, count_include_pad=True),
                attributes((3, 3), (1, 1), (1, 1, 1, 1), include_pad=True),
            ),
            Sample(
                "padded without pad",
                [x],
                lambda x: functional.avg_pool2d(x, 3, 1, 1, count_include_pad=False),
                attributes((3, 3), (1, 1), (1, 1, 1, 1), include_pad=False),
            ),
            Sample(
                "ceil",
                [x],
                lambda x: functional.avg_pool2d(x, 2, 2, ceil_mode=True),
                attributes((2, 2), (2, 2), ceil=True, include_pad=True),
            ),
        ],
    }


def attention_samples(pool, dtype):
    def attributes(scale, causal):
        return {"ScaledDotProductAttention": {"scale": scale, "causal": causal}}

    query = plain(pool, dtype, [1, 2, 3, 4])
    key = plain(pool, dtype, [1, 2, 5, 4], 2)
    value = plain(pool, dtype, [1, 2, 5, 3], 3)
    square_key = plain(pool, dtype, [1, 2, 3, 4], 2)
    square_value = plain(pool, dtype, [1, 2, 3, 3], 3)
    return (
        [
            Sample(
                "default scale",
                [query, key, value],
                lambda q, k, v: functional.scaled_dot_product_attention(q, k, v),
                attributes(None, False),
            ),
            Sample(
                "explicit scale",
                [query, key, value],
                lambda q, k, v: functional.scaled_dot_product_attention(
                    q, k, v, scale=0.5
                ),
                attributes(0.5, False),
            ),
            Sample(
                "causal",
                [query, square_key, square_value],
                lambda q, k, v: functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                ),
                attributes(None, True),
            ),
        ]
        + [
            Sample(
                label,
                inputs,
                lambda q, k, v, causal=causal, window=window, softcap=softcap: (
                    windowed_attention(q, k, v, causal, window, softcap)
                ),
                {
                    "ScaledDotProductAttention": {
                        "scale": None,
                        "causal": causal,
                        "window": window,
                        "softcap": softcap,
                    }
                },
            )
            for label, inputs, causal, window, softcap in [
                ("sliding window", [query, square_key, square_value], True, 2, None),
                ("softcap", [query, key, value], False, None, 0.5),
                (
                    "wide sliding window",
                    [
                        plain(pool, dtype, [1, 2, 70, 16]),
                        plain(pool, dtype, [1, 2, 70, 16], 2),
                        plain(pool, dtype, [1, 2, 70, 16], 3),
                    ],
                    True,
                    9,
                    1.5,
                ),
            ]
            # PyTorch's attention takes the types its own kernels compute, which the reference stays within.
            if dtype in FLOATS
        ]
        + [
            Sample(
                label,
                [
                    *inputs,
                    Input(
                        pool.explicit(
                            f"attention mask {'x'.join(map(str, shape))}",
                            attention_mask(dtype, shape),
                        )
                    ),
                ],
                lambda q, k, v, mask, causal=causal, softcap=softcap: (
                    windowed_attention(q, k, v, causal, None, softcap, mask)
                ),
                {
                    "ScaledDotProductAttention": {
                        "scale": None,
                        "causal": causal,
                        "window": None,
                        "softcap": softcap,
                        "masked": True,
                    }
                },
            )
            for label, inputs, shape, causal, softcap in [
                ("mask over heads", [query, key, value], [1, 1, 3, 5], False, None),
                ("mask per head", [query, key, value], [1, 2, 3, 5], False, None),
                (
                    "causal capped mask",
                    [query, square_key, square_value],
                    [3, 3],
                    True,
                    0.5,
                ),
                (
                    "wide mask",
                    [
                        plain(pool, dtype, [1, 2, 70, 16]),
                        plain(pool, dtype, [1, 2, 70, 16], 2),
                        plain(pool, dtype, [1, 2, 70, 16], 3),
                    ],
                    [1, 1, 70, 70],
                    False,
                    None,
                ),
            ]
            if dtype in FLOATS
        ]
        + [
            Sample(
                label,
                [*inputs, plain(pool, dtype, [heads], 5)],
                lambda q, k, v, sinks, causal=causal: windowed_attention(
                    q, k, v, causal, None, None, None, sinks
                ),
                {
                    "ScaledDotProductAttention": {
                        "scale": None,
                        "causal": causal,
                        "window": None,
                        "softcap": None,
                        "sinks": True,
                    }
                },
            )
            for label, inputs, heads, causal in [
                ("sinks", [query, key, value], 2, False),
                ("causal sinks", [query, square_key, square_value], 2, True),
                (
                    "wide sinks",
                    [
                        plain(pool, dtype, [1, 2, 70, 16]),
                        plain(pool, dtype, [1, 2, 70, 16], 2),
                        plain(pool, dtype, [1, 2, 70, 16], 3),
                    ],
                    2,
                    True,
                ),
            ]
            if dtype in FLOATS
        ]
    )


def attention_mask(dtype, shape):
    """An additive attention mask of `shape`: small biases, with every fifth key of each row hidden by negative infinity."""
    mask = values(dtype, shape, 4, "unit").to(torch.float64)
    keys = torch.arange(shape[-1])
    mask = mask.masked_fill(keys % 5 == 1, float("-inf"))
    return mask.to(DTYPES[dtype])


def windowed_attention(q, k, v, causal, window, softcap, mask=None, sinks=None):
    """Scaled dot-product attention in `F32`, or `F64` for `F64` operands, with an optional cap of `softcap · tanh(score / softcap)` on the scaled scores and a mask hiding keys after each query, when causal, and `window` or more positions before it. An additive `mask` adds to the capped scores, and `sinks`, one logit per head, join each row's softmax without weighing a value, as GPT-OSS writes them."""
    data_type = q.dtype
    wide = torch.float64 if data_type == torch.float64 else torch.float32
    q, k, v = q.to(wide), k.to(wide), v.to(wide)
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)
    if mask is not None:
        scores = scores + mask.to(wide)
    queries = torch.arange(q.shape[-2]).unsqueeze(-1)
    keys = torch.arange(k.shape[-2]).unsqueeze(0)
    hidden = torch.zeros(q.shape[-2], k.shape[-2], dtype=torch.bool)
    if causal:
        hidden |= keys > queries
    if window is not None:
        hidden |= keys + window <= queries
    scores = scores.masked_fill(hidden, float("-inf"))
    if sinks is not None:
        column = sinks.to(wide).reshape(1, -1, 1, 1).expand(*scores.shape[:-1], 1)
        weights = torch.softmax(torch.cat([scores, column], dim=-1), dim=-1)[..., :-1]
        return (weights @ v).to(data_type)
    return (torch.softmax(scores, dim=-1) @ v).to(data_type)


def paged_attention_samples(pool, dtype):
    """Ragged queries over a paged cache of pages of four slots, handed out out of order: a decoding query, a prefill chunk after cached keys, and a sequence with no keys."""
    if dtype not in FLOATS:
        return []
    query = plain(pool, dtype, [9, 4, 16])
    cache = [7, 4, 2, 16]
    key_cache, value_cache = plain(pool, dtype, cache, 2), plain(pool, dtype, cache, 3)
    index = lambda name, values: Input(
        pool.explicit(
            f"paged attention {name}", torch.tensor(values, dtype=torch.int32)
        )
    )
    indices = [
        index("query offsets", [0, 1, 6, 9]),
        index("page offsets", [0, 3, 5, 5]),
        index("pages", [4, 0, 6, 2, 5]),
        index("last page lengths", [2, 3, 0]),
    ]
    return [
        Sample(
            label,
            [query, key_cache, value_cache, *indices],
            lambda *operands, causal=causal: paged_attention_reference(
                *operands, causal
            ),
            {"ScaledDotProductAttention": {"scale": None, "causal": causal}},
        )
        for label, causal in [("ragged", False), ("ragged causal", True)]
    ]


def paged_attention_reference(
    q, k_cache, v_cache, query_offsets, page_offsets, pages, last_page_lengths, causal
):
    """Attention of each sequence's queries, its last tokens, over the keys of its pages, as FlashInfer-Bench's paged prefill writes it; a query that sees no key yields zeros."""
    data_type = q.dtype
    wide = torch.float64 if data_type == torch.float64 else torch.float32
    q, k_cache, v_cache = q.to(wide), k_cache.to(wide), v_cache.to(wide)
    _, page_size, kv_heads, dim = k_cache.shape
    groups = q.shape[1] // kv_heads
    output = torch.zeros_like(q)
    lse = torch.full(q.shape[:2], float("-inf"), dtype=torch.float32)
    for sequence in range(last_page_lengths.shape[0]):
        first, end = int(query_offsets[sequence]), int(query_offsets[sequence + 1])
        page_ids = pages[
            int(page_offsets[sequence]) : int(page_offsets[sequence + 1])
        ].long()
        keys = (
            0
            if len(page_ids) == 0
            else (len(page_ids) - 1) * page_size + int(last_page_lengths[sequence])
        )
        k = (
            k_cache[page_ids]
            .reshape(-1, kv_heads, dim)[:keys]
            .repeat_interleave(groups, dim=1)
        )
        v = (
            v_cache[page_ids]
            .reshape(-1, kv_heads, dim)[:keys]
            .repeat_interleave(groups, dim=1)
        )
        scores = torch.einsum("qhd,khd->hqk", q[first:end], k) / math.sqrt(dim)
        if causal:
            positions = torch.arange(end - first).unsqueeze(-1) + keys - (end - first)
            scores = scores.masked_fill(
                torch.arange(keys).unsqueeze(0) > positions, float("-inf")
            )
        if keys > 0 and end > first:
            lse[first:end] = (
                (torch.logsumexp(scores, dim=-1) / math.log(2.0))
                .transpose(0, 1)
                .to(torch.float32)
            )
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        output[first:end] = torch.einsum("hqk,khd->qhd", weights, v)
    return output.to(data_type), lse


def grouped_matmul_samples(pool, dtype):
    """Rows sorted by group times their group's weight, with an empty group, rows outside every group, and a strided weight."""
    if dtype not in FLOATS:
        return []
    rows = plain(pool, dtype, [9, 6])
    offsets = Input(
        pool.explicit(
            "grouped matmul offsets", torch.tensor([1, 4, 4, 8], dtype=torch.int32)
        )
    )
    return [
        Sample(
            "groups",
            [rows, plain(pool, dtype, [3, 6, 5], 2), offsets],
            grouped_matmul_reference,
        ),
        Sample(
            "strided weights",
            [rows, transposed(pool, dtype, [3, 6, 5]), offsets],
            grouped_matmul_reference,
        ),
    ]


def grouped_matmul_reference(rows, weights, offsets):
    """Each group's rows times its weight in `F32`, or `F64` for `F64` operands, with rows outside every group zero."""
    data_type = rows.dtype
    wide = torch.float64 if data_type == torch.float64 else torch.float32
    rows, weights = rows.to(wide), weights.to(wide)
    output = torch.zeros(rows.shape[0], weights.shape[2], dtype=wide)
    for group in range(weights.shape[0]):
        first, end = int(offsets[group]), int(offsets[group + 1])
        output[first:end] = rows[first:end] @ weights[group]
    return output.to(data_type)


# A step of four sequences over nine rows: three real rows from position zero and a padding row, one real row continuing at position five, a held row, two real rows continuing at position seven, and a padding row after every sequence.
STATE_SPANS = {
    "query offsets": [0, 4, 5, 6, 8],
    "real tokens": [3, 1, 0, 2],
    "positions": [0, 1, 2, 0, 5, 0, 7, 8, 0],
}


def state_span_inputs(pool, label):
    return [
        Input(pool.explicit(f"{label} {name}", torch.tensor(values, dtype=torch.int64)))
        for name, values in STATE_SPANS.items()
    ]


def state_sequences(offsets, real, positions):
    """Yields each sequence's index, first row, count of real rows, and whether it starts from a zero state."""
    for sequence in range(real.shape[0]):
        start, count = int(offsets[sequence]), int(real[sequence])
        yield sequence, start, count, count > 0 and int(positions[start]) == 0


def causal_conv1d_samples(pool, dtype):
    """A depthwise causal convolution of width four over the step's sequences, after their states."""
    if dtype not in FLOATS:
        return []
    rows, taps = plain(pool, dtype, [9, 6]), plain(pool, dtype, [6, 4], 2, "unit")
    state = plain(pool, dtype, [4, 6, 3], 3)
    spans = state_span_inputs(pool, "causal conv1d")
    return {
        "causal_conv1d": [
            Sample("sequences", [rows, taps, state, *spans], causal_conv1d_reference)
        ],
        "causal_conv1d_state": [
            Sample(
                "sequences",
                [rows, state, *spans],
                lambda rows, state, *spans: causal_conv1d_reference(
                    rows, None, state, *spans
                ),
            )
        ],
    }


def causal_conv1d_reference(rows, taps, state, offsets, real, positions):
    """Each sequence's real rows after its state, or zeros from position zero, convolved per channel as `transformers`' `torch_causal_conv1d_update` does, with the other rows zero; without taps, the states the step leaves."""
    data_type = rows.dtype
    wide = torch.float64 if data_type == torch.float64 else torch.float32
    rows, state = rows.to(wide), state.to(wide).clone()
    history = state.shape[-1]
    output = torch.zeros_like(rows)
    for sequence, start, count, fresh in state_sequences(offsets, real, positions):
        if count == 0:
            continue
        held = torch.zeros_like(state[sequence]) if fresh else state[sequence]
        window = torch.cat([held, rows[start : start + count].T], dim=1)
        if taps is not None:
            output[start : start + count] = functional.conv1d(
                window.unsqueeze(0), taps.to(wide).unsqueeze(1), groups=rows.shape[1]
            )[0].T
        state[sequence] = window[:, -history:]
    return (output if taps is not None else state).to(data_type)


def gated_delta_rule_samples(pool, dtype):
    """The gated delta rule over the step's sequences, with unit keys and queries, negative decays, and strengths between zero and one."""
    if dtype not in FLOATS:
        return []
    explicit = lambda name, tensor: Input(
        pool.explicit(f"gated delta rule {name}", tensor.to(DTYPES[dtype]))
    )
    unit = lambda seed: functional.normalize(
        values("F64", [9, 2, 8], seed, "any"), dim=-1
    )
    operands = [
        explicit("queries", unit(1)),
        explicit("keys", unit(2)),
        plain(pool, dtype, [9, 2, 6], 3),
        explicit("decays", -values("F64", [9, 2], 4, "positive") / 4),
        explicit("strengths", torch.sigmoid(values("F64", [9, 2], 5, "any"))),
        plain(pool, "F32", [4, 2, 8, 6], 6, "unit"),
        *state_span_inputs(pool, "gated delta rule"),
    ]
    return {
        name: [
            Sample(
                "sequences",
                operands,
                lambda *operands, want_state=want_state: gated_delta_rule_reference(
                    *operands, want_state
                ),
            )
        ]
        for name, want_state in [
            ("gated_delta_rule", False),
            ("gated_delta_rule_state", True),
        ]
    }


def gated_delta_rule_reference(
    q, k, v, g, beta, state, offsets, real, positions, want_state
):
    """Each sequence's real rows token by token from its state, or zeros from position zero, as `transformers`' `torch_recurrent_gated_delta_rule` runs them after its normalization, with the other rows zero; with `want_state`, the `F32` states the step leaves."""
    data_type = v.dtype
    wide = torch.float64 if data_type == torch.float64 else torch.float32
    q, k, v, g, beta = (tensor.to(wide) for tensor in (q, k, v, g, beta))
    states = state.to(wide).clone()
    output = torch.zeros_like(v)
    for sequence, start, count, fresh in state_sequences(offsets, real, positions):
        if count == 0:
            continue
        held = torch.zeros_like(states[sequence]) if fresh else states[sequence]
        for row in range(start, start + count):
            held = held * g[row].exp()[:, None, None]
            read = (held * k[row][:, :, None]).sum(dim=-2)
            delta = (v[row] - read) * beta[row][:, None]
            held = held + k[row][:, :, None] * delta[:, None, :]
            output[row] = (held * q[row][:, :, None]).sum(dim=-2)
        states[sequence] = held
    return states.to(torch.float32) if want_state else output.to(data_type)


def paged_mla_attention_samples(pool, dtype):
    """Absorbed latent queries over a paged latent cache of pages of three slots, handed out out of order: a decoding query, a prefill chunk after cached keys, and a sequence with no keys."""
    if dtype not in FLOATS:
        return []
    latent_query, positional_query = (
        plain(pool, dtype, [9, 4, 32]),
        plain(pool, dtype, [9, 4, 16], 2),
    )
    latent_cache, positional_cache = (
        plain(pool, dtype, [8, 3, 32], 3),
        plain(pool, dtype, [8, 3, 16], 4),
    )
    index = lambda name, values: Input(
        pool.explicit(
            f"paged mla attention {name}", torch.tensor(values, dtype=torch.int32)
        )
    )
    indices = [
        index("query offsets", [0, 1, 6, 9]),
        index("page offsets", [0, 3, 6, 6]),
        index("pages", [4, 0, 7, 2, 5, 1]),
        index("last page lengths", [2, 3, 0]),
    ]
    return [
        Sample(
            label,
            [latent_query, positional_query, latent_cache, positional_cache, *indices],
            lambda *operands, causal=causal: paged_mla_attention_reference(
                *operands, causal
            ),
            {"ScaledDotProductAttention": {"scale": None, "causal": causal}},
        )
        for label, causal in [("ragged", False), ("ragged causal", True)]
    ]


def paged_mla_attention_reference(
    q_nope,
    q_pe,
    ckv_cache,
    kpe_cache,
    query_offsets,
    page_offsets,
    pages,
    last_page_lengths,
    causal,
):
    """Latent attention as FlashInfer-Bench's MLA decode and prefill write it: scores of latent and positional parts over the shared caches, outputs weighing the latent cache, and each row's base-two log-sum-exp, negative infinity where a row sees no key."""
    data_type = q_nope.dtype
    wide = torch.float64 if data_type == torch.float64 else torch.float32
    q_nope, q_pe, ckv_cache, kpe_cache = (
        tensor.to(wide) for tensor in (q_nope, q_pe, ckv_cache, kpe_cache)
    )
    page_size = ckv_cache.shape[1]
    scale = 1.0 / math.sqrt(q_nope.shape[-1] + q_pe.shape[-1])
    output = torch.zeros_like(q_nope)
    lse = torch.full(q_nope.shape[:2], float("-inf"), dtype=torch.float32)
    for sequence in range(last_page_lengths.shape[0]):
        first, end = int(query_offsets[sequence]), int(query_offsets[sequence + 1])
        page_ids = pages[
            int(page_offsets[sequence]) : int(page_offsets[sequence + 1])
        ].long()
        keys = (
            0
            if len(page_ids) == 0
            else (len(page_ids) - 1) * page_size + int(last_page_lengths[sequence])
        )
        if keys == 0 or end == first:
            continue
        kc = ckv_cache[page_ids].reshape(-1, ckv_cache.shape[-1])[:keys]
        kp = kpe_cache[page_ids].reshape(-1, kpe_cache.shape[-1])[:keys]
        scores = (
            torch.einsum("qhd,kd->qhk", q_nope[first:end], kc)
            + torch.einsum("qhd,kd->qhk", q_pe[first:end], kp)
        ) * scale
        if causal:
            positions = (
                torch.arange(end - first).reshape(-1, 1, 1) + keys - (end - first)
            )
            scores = scores.masked_fill(
                torch.arange(keys).reshape(1, 1, -1) > positions, float("-inf")
            )
        lse[first:end] = (torch.logsumexp(scores, dim=-1) / math.log(2.0)).to(
            torch.float32
        )
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        output[first:end] = torch.einsum("qhk,kd->qhd", weights, kc)
    return output.to(data_type), lse


def rotary_reference(x, cos, sin):
    """Rotates each adjacent pair of the leading `2 * cos.shape[1]` features by its position's angle, as ONNX's interleaved rotary embedding does."""
    pairs = cos.shape[1]
    rotated = x[..., : 2 * pairs].unflatten(-1, (pairs, 2))
    even, odd = rotated[..., 0], rotated[..., 1]
    result = torch.stack(
        [even * cos - odd * sin, even * sin + odd * cos], dim=-1
    ).flatten(-2)
    return torch.cat([result, x[..., 2 * pairs :]], dim=-1)


def rotary_samples(pool, dtype):
    x = plain(pool, dtype, [1, 2, 3, 8])
    return [
        Sample(
            "full",
            [x, plain(pool, dtype, [3, 4], 2), plain(pool, dtype, [3, 4], 3)],
            rotary_reference,
        ),
        Sample(
            "partial",
            [x, plain(pool, dtype, [3, 2], 2), plain(pool, dtype, [3, 2], 3)],
            rotary_reference,
        ),
    ]


def scaled_matmul_samples(pool, dtype):
    """`torch._scaled_mm` of FP8 operands: a row-major left operand by a column-major right one, as cuBLASLt takes them, and shapes and scales it does not take."""

    def scale(name, values, shape):
        return Input(
            pool.explicit(
                f"scaled_mm {name}",
                torch.tensor(values, dtype=torch.float32).reshape(shape),
            )
        )

    def sample(label, lhs, rhs, scales, output):
        return Sample(
            label,
            [lhs, rhs, *scales],
            lambda a, b, a_scale, b_scale, output=output: (
                (a.float() * a_scale) @ (b.float() * b_scale)
            ).to(DTYPES[output]),
            {"Cast": output},
        )

    tensorwise = [
        scale("tensor scale", [0.5], [1]),
        scale("other tensor scale", [1.5], [1]),
    ]
    rowwise = [
        scale("row scales", [0.5 + row / 8 for row in range(16)], [16, 1]),
        scale("column scales", [1.0 + column / 16 for column in range(32)], [1, 32]),
    ]
    aligned = [plain(pool, dtype, [16, 32]), transposed(pool, dtype, [32, 32], 2)]
    return [
        sample("aligned into Bf16", *aligned, tensorwise, "Bf16"),
        sample("aligned into F32", *aligned, tensorwise, "F32"),
        sample("row-wise scales", *aligned, rowwise, "F16"),
        sample(
            "unaligned",
            plain(pool, dtype, [5, 24]),
            plain(pool, dtype, [24, 7], 2),
            tensorwise,
            "F32",
        ),
    ]


def quantization_samples(pool):
    x = plain(pool, "F32", [3, 4])
    scale = Input(pool.explicit("scale 0.05", torch.tensor(0.05)))
    zero_point = Input(pool.explicit("zero point 3", torch.tensor(3)))
    channel_scales = Input(
        pool.explicit("scales [0.05,0.1,0.02]", torch.tensor([0.05, 0.1, 0.02]))
    )
    channel_zero_points = Input(
        pool.explicit("zero points [0,-2,5]", torch.tensor([0, -2, 5]))
    )
    special_values = Input(pool.special("F32"))

    def attributes(data_type=None, axis=None):
        return {
            "Quantization": {
                "axis": axis,
                "block_size": None,
                "data_type": data_type,
                "packed": None,
            }
        }

    def quantize(dtype, axis=None):
        def reference(x, scale, zero_point):
            if axis is None:
                q = torch.round(x / scale) + zero_point
            else:
                shape = [1] * x.dim()
                shape[axis] = -1
                q = torch.round(x / scale.reshape(shape)) + zero_point.reshape(shape)
            info = torch.iinfo(DTYPES[dtype])
            return (
                torch.nan_to_num(q, nan=float(zero_point.flatten()[0]))
                .clamp(info.min, info.max)
                .to(DTYPES[dtype])
            )

        return reference

    def dequantize(x, scale, zero_point):
        return (x.to(torch.float32) - zero_point.to(torch.float32)) * scale

    def quantized_matmul(a, b, a_scale, a_zero, b_scale, b_zero):
        return dequantize(a, a_scale, a_zero) @ dequantize(b, b_scale, b_zero)

    samples = {"quantize": [], "dequantize": []}
    for dtype in ["I8", "U8"]:
        samples["quantize"] += [
            Sample(
                f"{dtype} tensor",
                [x, scale, zero_point],
                quantize(dtype),
                attributes(dtype),
            ),
            Sample(
                f"{dtype} special",
                [special_values, scale, zero_point],
                quantize(dtype),
                attributes(dtype),
            ),
            Sample(
                f"{dtype} channels",
                [plain(pool, "F32", [3, 4]), channel_scales, channel_zero_points],
                quantize(dtype, 0),
                attributes(dtype, 0),
            ),
        ]
        samples["dequantize"] += [
            Sample(
                f"{dtype} tensor",
                [plain(pool, dtype, [3, 4]), scale, zero_point],
                dequantize,
                attributes(),
            ),
        ]
    samples["quantized_matmul"] = [
        Sample(
            "I8",
            [
                plain(pool, "I8", [3, 4]),
                plain(pool, "I8", [4, 5], 2),
                scale,
                zero_point,
                scale,
                zero_point,
            ],
            quantized_matmul,
        ),
    ]
    return samples


def cast_samples(pool, source, target):
    """Returns the samples of a cast from `source` to `target`, avoiding float values PyTorch converts to integers without a defined result."""
    attributes = {"Cast": target}

    def reference(x):
        return x.to(DTYPES[target])

    integer_target = target in SIGNED or target in UNSIGNED
    if is_float(source) and integer_target:
        values = [0.0, -0.0, 0.5, 1.5, 2.75, 7.0, 99.9, 3.25, 0.25, 1.0, 5.5, 6.0]
        if target in SIGNED:
            values = [value * (-1) ** index for index, value in enumerate(values)]
        rounding = pool.explicit(
            f"rounding to {target}",
            torch.tensor(values, dtype=torch.float64).to(DTYPES[source]),
        )
        return [
            Sample(
                "matrix",
                [plain(pool, source, [3, 4], 1, "positive")],
                reference,
                attributes,
            ),
            Sample("rounding", [Input(rounding)], reference, attributes),
        ]
    samples = [
        Sample("matrix", [plain(pool, source, [3, 4])], reference, attributes),
        Sample("transposed", [transposed(pool, source, [3, 4])], reference, attributes),
    ]
    if source != "Bool":
        samples.append(
            Sample("special", [Input(pool.special(source))], reference, attributes)
        )
    return samples


def grouped_topk_routing_reference(logits, bias, groups, kept_groups, top_k, scale):
    """DeepSeek-V3's no-aux routing as FlashInfer-Bench's MoE reference writes it, with the chosen experts sorted by biased score."""
    tokens, experts = logits.shape
    scores = torch.sigmoid(logits.to(torch.float32))
    biased = scores + bias.to(torch.float32)
    best, _ = torch.topk(biased.view(tokens, groups, experts // groups), k=2, dim=2)
    _, kept = torch.topk(best.sum(dim=2), k=kept_groups, dim=1)
    mask = torch.zeros(tokens, groups).scatter_(1, kept, 1.0)
    mask = (
        mask.unsqueeze(2)
        .expand(tokens, groups, experts // groups)
        .reshape(tokens, experts)
    )
    pruned = biased.masked_fill(mask == 0, torch.finfo(torch.float32).min)
    _, ids = torch.topk(pruned, k=top_k, dim=1, sorted=True)
    chosen = scores.gather(1, ids)
    return ids, chosen / (chosen.sum(dim=1, keepdim=True) + 1e-20) * scale


def composite_samples(pool, dtype):
    """Returns samples of the tensor methods Singe composes from registry operations, whose attributes name the method's arguments."""
    matrix = plain(pool, dtype, [3, 4])
    rank3 = plain(pool, dtype, [2, 3, 4])

    def conv(stride=1, padding=(0, 0), dilation=1, groups=1):
        def reference(x, w, b=None):
            return functional.conv1d(
                functional.pad(x, padding), w, b, stride, 0, dilation, groups
            )

        return reference

    def conv_arguments(stride=1, padding=(0, 0), dilation=1, groups=1):
        return {
            "stride": stride,
            "padding": list(padding),
            "dilation": dilation,
            "groups": groups,
        }

    samples = {
        "select": [
            Sample("rows", [matrix], lambda x: x.select(0, 1), {"dim": 0, "index": 1}),
            Sample(
                "negative",
                [matrix],
                lambda x: x.select(-1, -1),
                {"dim": -1, "index": -1},
            ),
            Sample("rank3", [rank3], lambda x: x.select(1, 2), {"dim": 1, "index": 2}),
            Sample(
                "out of range",
                [matrix],
                lambda x: x.select(0, 3),
                {"dim": 0, "index": 3},
            ),
        ],
        "nonzero": [
            Sample("matrix", [matrix], torch.nonzero, {}),
            Sample("scalar", [plain(pool, dtype, [])], torch.nonzero, {}),
            Sample("empty", [plain(pool, dtype, [0, 4])], torch.nonzero, {}),
            Sample("transposed", [transposed(pool, dtype, [3, 4])], torch.nonzero, {}),
        ],
        "masked_select": [
            Sample(
                "matrix",
                [matrix, plain(pool, "Bool", [3, 4], 3)],
                torch.masked_select,
                {},
            ),
            Sample(
                "broadcast mask",
                [matrix, plain(pool, "Bool", [4], 3)],
                torch.masked_select,
                {},
            ),
            Sample(
                "broadcast values",
                [plain(pool, dtype, [4]), plain(pool, "Bool", [3, 4], 3)],
                torch.masked_select,
                {},
            ),
        ],
        "flip": [
            Sample("rows", [matrix], lambda x: torch.flip(x, [0]), {"dims": [0]}),
            Sample(
                "both", [matrix], lambda x: torch.flip(x, [0, -1]), {"dims": [0, -1]}
            ),
            Sample(
                "transposed",
                [transposed(pool, dtype, [3, 4])],
                lambda x: torch.flip(x, [1]),
                {"dims": [1]},
            ),
            Sample("none", [rank3], lambda x: torch.flip(x, []), {"dims": []}),
        ],
        "roll": [
            Sample(
                "rows",
                [matrix],
                lambda x: torch.roll(x, 1, 0),
                {"shifts": [1], "dims": [0]},
            ),
            Sample(
                "negative",
                [matrix],
                lambda x: torch.roll(x, -5, 1),
                {"shifts": [-5], "dims": [1]},
            ),
            Sample(
                "two",
                [rank3],
                lambda x: torch.roll(x, (2, 1), (1, 2)),
                {"shifts": [2, 1], "dims": [1, 2]},
            ),
            Sample(
                "flattened",
                [matrix],
                lambda x: torch.roll(x, 3),
                {"shifts": [3], "dims": []},
            ),
        ],
        "unfold": [
            Sample(
                "windows",
                [plain(pool, dtype, [7])],
                lambda x: x.unfold(0, 3, 2),
                {"dim": 0, "size": 3, "step": 2},
            ),
            Sample(
                "rows",
                [matrix],
                lambda x: x.unfold(1, 2, 1),
                {"dim": 1, "size": 2, "step": 1},
            ),
            Sample(
                "transposed",
                [transposed(pool, dtype, [3, 4])],
                lambda x: x.unfold(0, 2, 1),
                {"dim": 0, "size": 2, "step": 1},
            ),
        ],
        "diagonal": [
            Sample(
                "main",
                [matrix],
                lambda x: torch.diagonal(x),
                {"offset": 0, "dim1": 0, "dim2": 1},
            ),
            Sample(
                "above",
                [matrix],
                lambda x: torch.diagonal(x, 1),
                {"offset": 1, "dim1": 0, "dim2": 1},
            ),
            Sample(
                "below",
                [matrix],
                lambda x: torch.diagonal(x, -1),
                {"offset": -1, "dim1": 0, "dim2": 1},
            ),
            Sample(
                "rank3",
                [rank3],
                lambda x: torch.diagonal(x, 0, 1, 2),
                {"offset": 0, "dim1": 1, "dim2": 2},
            ),
            Sample(
                "empty",
                [matrix],
                lambda x: torch.diagonal(x, 5),
                {"offset": 5, "dim1": 0, "dim2": 1},
            ),
        ],
        "unbind": [
            Sample("rows", [matrix], lambda x: torch.unbind(x, 0), {"dim": 0}),
            Sample(
                "columns",
                [transposed(pool, dtype, [3, 4])],
                lambda x: torch.unbind(x, -1),
                {"dim": -1},
            ),
        ],
        "split": [
            Sample(
                "even", [matrix], lambda x: torch.split(x, 2, 1), {"size": 2, "dim": 1}
            ),
            Sample(
                "uneven",
                [plain(pool, dtype, [7])],
                lambda x: torch.split(x, 3),
                {"size": 3, "dim": 0},
            ),
        ],
        "chunk": [
            Sample(
                "even",
                [matrix],
                lambda x: torch.chunk(x, 2, 1),
                {"chunks": 2, "dim": 1},
            ),
            Sample(
                "uneven",
                [plain(pool, dtype, [7])],
                lambda x: torch.chunk(x, 3),
                {"chunks": 3, "dim": 0},
            ),
            Sample(
                "more than elements",
                [plain(pool, dtype, [2])],
                lambda x: torch.chunk(x, 5),
                {"chunks": 5, "dim": 0},
            ),
        ],
        "any": [
            Sample(
                "rows",
                [matrix],
                lambda x: torch.any(x, 1),
                {"dims": [1], "keepdim": False},
            ),
            Sample("all axes", [matrix], torch.any, {"dims": [], "keepdim": False}),
            Sample(
                "empty",
                [plain(pool, dtype, [0, 4])],
                lambda x: torch.any(x, 0, keepdim=True),
                {"dims": [0], "keepdim": True},
            ),
        ],
        "all": [
            Sample(
                "rows",
                [matrix],
                lambda x: torch.all(x, 1),
                {"dims": [1], "keepdim": False},
            ),
            Sample("all axes", [matrix], torch.all, {"dims": [], "keepdim": False}),
            Sample(
                "empty",
                [plain(pool, dtype, [0, 4])],
                lambda x: torch.all(x, 0, keepdim=True),
                {"dims": [0], "keepdim": True},
            ),
        ],
        "repeat": [
            Sample("rows", [matrix], lambda x: x.repeat(2, 1), {"repeats": [2, 1]}),
            Sample(
                "new axis",
                [plain(pool, dtype, [2, 3])],
                lambda x: x.repeat(2, 1, 2),
                {"repeats": [2, 1, 2]},
            ),
            Sample(
                "scalar",
                [plain(pool, dtype, [])],
                lambda x: x.repeat(3),
                {"repeats": [3]},
            ),
            Sample(
                "transposed",
                [transposed(pool, dtype, [3, 4])],
                lambda x: x.repeat(1, 2),
                {"repeats": [1, 2]},
            ),
        ],
    }
    if dtype in FLOATS or dtype in SIGNED:
        samples["addmm"] = [
            Sample(
                "matrix",
                [plain(pool, dtype, [3, 5], 3), matrix, plain(pool, dtype, [4, 5], 2)],
                lambda x, a, b: torch.addmm(x, a, b),
                {"beta": 1, "alpha": 1},
            ),
            Sample(
                "scaled bias",
                [plain(pool, dtype, [5], 3), matrix, plain(pool, dtype, [4, 5], 2)],
                lambda x, a, b: torch.addmm(x, a, b, beta=2, alpha=3),
                {"beta": 2, "alpha": 3},
            ),
            Sample(
                "no input",
                [plain(pool, dtype, [3, 5], 3), matrix, plain(pool, dtype, [4, 5], 2)],
                lambda x, a, b: torch.addmm(x, a, b, beta=0),
                {"beta": 0, "alpha": 1},
            ),
        ]
    if dtype in FLOATS:
        x = plain(pool, dtype, [1, 2, 7])
        weight = plain(pool, dtype, [3, 2, 3], 2)
        bias = plain(pool, dtype, [3], 3)
        samples["conv1d"] = [
            Sample(
                "padded",
                [x, weight, bias],
                conv(padding=(1, 1)),
                conv_arguments(padding=(1, 1)),
            ),
            Sample("strided", [x, weight], conv(stride=2), conv_arguments(stride=2)),
            Sample(
                "dilated",
                [x, weight, bias],
                conv(dilation=2),
                conv_arguments(dilation=2),
            ),
            Sample(
                "asymmetric padding",
                [x, weight],
                conv(padding=(2, 0)),
                conv_arguments(padding=(2, 0)),
            ),
            Sample(
                "grouped",
                [plain(pool, dtype, [1, 4, 6]), plain(pool, dtype, [2, 2, 3], 2)],
                conv(groups=2),
                conv_arguments(groups=2),
            ),
        ]
    if dtype in FLOATS:
        wide = plain(pool, dtype, [3, 5])
        specials = Input(pool.special(dtype))

        def activation(function, arguments):
            return [
                Sample("matrix", [wide], function, arguments),
                Sample("special", [specials], function, arguments),
                Sample(
                    "transposed", [transposed(pool, dtype, [3, 4])], function, arguments
                ),
            ]

        samples["leaky_relu"] = activation(
            lambda x: functional.leaky_relu(x, 0.2), {"negative_slope": 0.2}
        )
        samples["elu"] = activation(lambda x: functional.elu(x, 1.5), {"alpha": 1.5})
        samples["selu"] = activation(functional.selu, {})
        samples["celu"] = activation(lambda x: functional.celu(x, 0.7), {"alpha": 0.7})
        samples["softsign"] = activation(functional.softsign, {})
        samples["hardsigmoid"] = activation(functional.hardsigmoid, {})
        samples["hardswish"] = activation(functional.hardswish, {})
        samples["hardtanh"] = activation(
            lambda x: functional.hardtanh(x, -0.5, 1.5), {"min": -0.5, "max": 1.5}
        )
        samples["softplus"] = activation(
            lambda x: functional.softplus(x, 2.0, 1e9), {"beta": 2.0}
        )
        samples["mish"] = activation(functional.mish, {})
        samples["logsumexp"] = [
            Sample(
                "rows",
                [wide],
                lambda x: torch.logsumexp(x, [1]),
                {"dims": [1], "keepdim": False},
            ),
            Sample(
                "kept",
                [rank3],
                lambda x: torch.logsumexp(x, [0, 2], keepdim=True),
                {"dims": [0, 2], "keepdim": True},
            ),
            Sample(
                "special",
                [specials],
                lambda x: torch.logsumexp(x, [0]),
                {"dims": [0], "keepdim": False},
            ),
        ]
        samples["var"] = [
            Sample(
                "rows",
                [wide],
                lambda x: torch.var(x, [1]),
                {"dims": [1], "correction": 1, "keepdim": False},
            ),
            Sample(
                "population",
                [rank3],
                lambda x: torch.var(x, [0, 2], correction=0, keepdim=True),
                {"dims": [0, 2], "correction": 0, "keepdim": True},
            ),
        ]
        samples["std"] = [
            Sample(
                "rows",
                [wide],
                lambda x: torch.std(x, [1]),
                {"dims": [1], "correction": 1, "keepdim": False},
            ),
        ]
        samples["vector_norm"] = [
            Sample(
                "one",
                [wide],
                lambda x: torch.linalg.vector_norm(x, 1, [1]),
                {"order": 1.0, "dims": [1], "keepdim": False},
            ),
            Sample(
                "two",
                [rank3],
                lambda x: torch.linalg.vector_norm(x, 2, [0, 2], keepdim=True),
                {"order": 2.0, "dims": [0, 2], "keepdim": True},
            ),
            Sample(
                "three",
                [wide],
                lambda x: torch.linalg.vector_norm(x, 3, [0]),
                {"order": 3.0, "dims": [0], "keepdim": False},
            ),
            Sample(
                "infinity",
                [wide],
                lambda x: torch.linalg.vector_norm(x, math.inf, [1]),
                {"order": "inf", "dims": [1], "keepdim": False},
            ),
            Sample(
                "negative infinity",
                [wide],
                lambda x: torch.linalg.vector_norm(x, -math.inf, [1]),
                {"order": "-inf", "dims": [1], "keepdim": False},
            ),
        ]
    samples["stack"] = [
        Sample(
            "rows",
            [matrix, plain(pool, dtype, [3, 4], 2)],
            lambda a, b: torch.stack([a, b], 0),
            {"dim": 0},
        ),
        Sample(
            "last",
            [matrix, plain(pool, dtype, [3, 4], 2)],
            lambda a, b: torch.stack([a, b], -1),
            {"dim": -1},
        ),
    ]
    samples["triu"] = [
        Sample("main", [matrix], torch.triu, {"diagonal": 0}),
        Sample("above", [rank3], lambda x: torch.triu(x, 1), {"diagonal": 1}),
        Sample(
            "below",
            [transposed(pool, dtype, [3, 4])],
            lambda x: torch.triu(x, -1),
            {"diagonal": -1},
        ),
    ]
    samples["tril"] = [
        Sample("main", [matrix], torch.tril, {"diagonal": 0}),
        Sample("above", [rank3], lambda x: torch.tril(x, 2), {"diagonal": 2}),
    ]
    samples["repeat_interleave"] = [
        Sample(
            "rows",
            [matrix],
            lambda x: torch.repeat_interleave(x, 2, 0),
            {"repeats": 2, "dim": 0},
        ),
        Sample(
            "last",
            [rank3],
            lambda x: torch.repeat_interleave(x, 3, -1),
            {"repeats": 3, "dim": -1},
        ),
    ]
    if dtype in ("F32", "F64"):
        samples["nan_to_num"] = [
            Sample(
                "defaults", [Input(pool.special(dtype))], torch.nan_to_num, {"nan": 0.0}
            ),
            Sample(
                "chosen",
                [Input(pool.special(dtype))],
                lambda x: torch.nan_to_num(x, 1.5, 100.0, -200.0),
                {"nan": 1.5, "posinf": 100.0, "neginf": -200.0},
            ),
        ]
    samples["sign"] = [
        Sample("matrix", [matrix], torch.sign, {}),
        Sample("special", [Input(pool.special(dtype))], torch.sign, {}),
    ]
    samples["movedim"] = [
        Sample(
            "forward",
            [rank3],
            lambda x: torch.movedim(x, 0, 2),
            {"source": 0, "destination": 2},
        ),
        Sample(
            "backward",
            [rank3],
            lambda x: torch.movedim(x, -1, 0),
            {"source": -1, "destination": 0},
        ),
    ]
    if dtype != "Bool":
        samples["outer"] = [
            Sample(
                "vectors",
                [plain(pool, dtype, [3]), plain(pool, dtype, [4], 2)],
                torch.outer,
                {},
            ),
        ]
    if dtype in FLOATS or dtype in SIGNED:
        samples["masked_fill"] = [
            Sample(
                "matrix",
                [matrix, plain(pool, "Bool", [3, 4], 3)],
                lambda x, m: x.masked_fill(m, 2),
                {"value": 2},
            ),
            Sample(
                "broadcast mask",
                [matrix, plain(pool, "Bool", [4], 3)],
                lambda x, m: x.masked_fill(m, -1),
                {"value": -1},
            ),
        ]
    if dtype in ("F32", "F64"):

        def einsum(equation, *shapes):
            inputs = [
                plain(pool, dtype, shape, seed + 1) for seed, shape in enumerate(shapes)
            ]
            return Sample(
                equation,
                inputs,
                lambda *operands: torch.einsum(equation, *operands),
                {"equation": equation},
            )

        samples["einsum"] = [
            einsum("ij,jk->ik", [3, 4], [4, 5]),
            einsum("bij,bjk->bik", [2, 3, 4], [2, 4, 5]),
            einsum("bhqd,bhkd->bhqk", [2, 2, 3, 4], [2, 2, 5, 4]),
            einsum("ij->ji", [3, 4]),
            einsum("ij->", [3, 4]),
            einsum("i,j->ij", [3], [4]),
            einsum("ij,jk,kl->il", [2, 3], [3, 4], [4, 2]),
            einsum("ij,kj", [3, 4], [5, 4]),
            einsum("ijk,ik->j", [2, 3, 4], [2, 4]),
            einsum("ii->i", [3, 3]),
            einsum("ii", [3, 3]),
            einsum("iij->j", [3, 3, 2]),
            einsum("...ij->...ji", [2, 3, 4]),
            einsum("...ij,...jk->...ik", [2, 3, 4], [4, 5]),
            einsum("...ij,...jk", [3, 1, 3, 4], [2, 4, 5]),
            einsum("i...,i...->...", [3, 2], [3, 1, 2]),
            einsum("...ij->j", [2, 3, 4]),
            einsum("ij,j->i", [3, 1], [5]),
            einsum("bii,bj->bj", [2, 3, 3], [2, 4]),
        ]
        image = plain(pool, dtype, [2, 4, 3, 3])
        weight = plain(pool, dtype, [4], 2)
        bias = plain(pool, dtype, [4], 3)
        samples["group_norm"] = [
            Sample(
                "two groups",
                [image, weight, bias],
                lambda x, w, b: functional.group_norm(x, 2, w, b, 1e-5),
                {"groups": 2, "eps": 1e-5},
            ),
            Sample(
                "one group",
                [image],
                lambda x: functional.group_norm(x, 1, eps=1e-5),
                {"groups": 1, "eps": 1e-5},
            ),
        ]
        samples["instance_norm"] = [
            Sample(
                "affine",
                [image, weight, bias],
                lambda x, w, b: functional.instance_norm(x, weight=w, bias=b, eps=1e-5),
                {"eps": 1e-5},
            ),
        ]
        sequence = plain(pool, dtype, [2, 3, 9])
        samples["max_pool1d"] = [
            Sample(
                "strided",
                [sequence],
                lambda x: functional.max_pool1d(x, 3, 2),
                {"kernel": 3, "stride": 2, "padding": [0, 0]},
            ),
            Sample(
                "padded",
                [sequence],
                lambda x: functional.max_pool1d(x, 2, 1, 1),
                {"kernel": 2, "stride": 1, "padding": [1, 1]},
            ),
        ]
        samples["avg_pool1d"] = [
            Sample(
                "strided",
                [sequence],
                lambda x: functional.avg_pool1d(x, 3, 2),
                {"kernel": 3, "stride": 2, "padding": [0, 0]},
            ),
            Sample(
                "padded",
                [sequence],
                lambda x: functional.avg_pool1d(x, 2, 1, 1),
                {"kernel": 2, "stride": 1, "padding": [1, 1]},
            ),
        ]
        picture = plain(pool, dtype, [1, 2, 5, 7])
        samples["adaptive_avg_pool2d"] = [
            Sample(
                f"{h}x{w}",
                [picture],
                lambda x, h=h, w=w: functional.adaptive_avg_pool2d(x, (h, w)),
                {"size": [h, w]},
            )
            for h, w in ((1, 1), (3, 2), (5, 7), (2, 4))
        ]
        samples["adaptive_max_pool2d"] = [
            Sample(
                f"{h}x{w}",
                [picture],
                lambda x, h=h, w=w: functional.adaptive_max_pool2d(x, (h, w)),
                {"size": [h, w]},
            )
            for h, w in ((1, 1), (3, 2), (2, 4))
        ]
        volume = plain(pool, dtype, [1, 2, 5, 6, 7])
        samples["max_pool3d"] = [
            Sample(
                "cube",
                [volume],
                lambda x: functional.max_pool3d(x, 2, 2),
                {
                    "kernel": [2, 2, 2],
                    "stride": [2, 2, 2],
                    "padding": [0, 0, 0],
                    "ceil": False,
                },
            ),
            Sample(
                "padded",
                [volume],
                lambda x: functional.max_pool3d(x, (3, 2, 3), (1, 2, 2), (1, 0, 1)),
                {
                    "kernel": [3, 2, 3],
                    "stride": [1, 2, 2],
                    "padding": [1, 0, 1],
                    "ceil": False,
                },
            ),
            Sample(
                "ceil",
                [volume],
                lambda x: functional.max_pool3d(x, 2, 2, ceil_mode=True),
                {
                    "kernel": [2, 2, 2],
                    "stride": [2, 2, 2],
                    "padding": [0, 0, 0],
                    "ceil": True,
                },
            ),
        ]
        samples["avg_pool3d"] = [
            Sample(
                "cube",
                [volume],
                lambda x: functional.avg_pool3d(x, 2, 2),
                {
                    "kernel": [2, 2, 2],
                    "stride": [2, 2, 2],
                    "padding": [0, 0, 0],
                    "ceil": False,
                    "include_pad": True,
                },
            ),
            Sample(
                "padded",
                [volume],
                lambda x: functional.avg_pool3d(x, (3, 2, 3), (1, 2, 2), (1, 0, 1)),
                {
                    "kernel": [3, 2, 3],
                    "stride": [1, 2, 2],
                    "padding": [1, 0, 1],
                    "ceil": False,
                    "include_pad": True,
                },
            ),
            Sample(
                "excluding padding",
                [volume],
                lambda x: functional.avg_pool3d(x, 3, 1, 1, count_include_pad=False),
                {
                    "kernel": [3, 3, 3],
                    "stride": [1, 1, 1],
                    "padding": [1, 1, 1],
                    "ceil": False,
                    "include_pad": False,
                },
            ),
        ]
        voxels = plain(pool, dtype, [1, 4, 5, 6, 5])
        samples["conv3d"] = [
            Sample(
                "padded",
                [
                    voxels,
                    plain(pool, dtype, [3, 4, 3, 3, 3], 2),
                    plain(pool, dtype, [3], 3),
                ],
                lambda x, w, b: functional.conv3d(x, w, b, (1, 1, 1), (1, 1, 1)),
                {
                    "stride": [1, 1, 1],
                    "padding": [1, 1, 1],
                    "dilation": [1, 1, 1],
                    "groups": 1,
                },
            ),
            Sample(
                "strided groups",
                [voxels, plain(pool, dtype, [4, 2, 2, 3, 2], 2)],
                lambda x, w: functional.conv3d(x, w, None, (2, 1, 2), (0, 1, 0), 1, 2),
                {
                    "stride": [2, 1, 2],
                    "padding": [0, 1, 0],
                    "dilation": [1, 1, 1],
                    "groups": 2,
                },
            ),
            Sample(
                "dilated",
                [voxels, plain(pool, dtype, [2, 4, 2, 2, 2], 2)],
                lambda x, w: functional.conv3d(x, w, None, 1, 0, (2, 1, 2)),
                {
                    "stride": [1, 1, 1],
                    "padding": [0, 0, 0],
                    "dilation": [2, 1, 2],
                    "groups": 1,
                },
            ),
        ]
        image4 = plain(pool, dtype, [1, 4, 5, 6])

        def transpose_arguments(stride, padding, output_padding, dilation, groups):
            return {
                "stride": list(stride),
                "padding": list(padding),
                "output_padding": list(output_padding),
                "dilation": list(dilation),
                "groups": groups,
            }

        samples["conv_transpose2d"] = [
            Sample(
                "strided",
                [
                    image4,
                    plain(pool, dtype, [4, 3, 3, 3], 2),
                    plain(pool, dtype, [3], 3),
                ],
                lambda x, w, b: functional.conv_transpose2d(
                    x, w, b, (2, 2), (1, 1), (1, 0)
                ),
                transpose_arguments((2, 2), (1, 1), (1, 0), (1, 1), 1),
            ),
            Sample(
                "grouped dilated",
                [image4, plain(pool, dtype, [4, 2, 2, 3], 2)],
                lambda x, w: functional.conv_transpose2d(
                    x, w, None, (1, 3), (0, 2), (0, 1), 2, (2, 1)
                ),
                transpose_arguments((1, 3), (0, 2), (0, 1), (2, 1), 2),
            ),
            Sample(
                "cropping",
                [image4, plain(pool, dtype, [4, 2, 2, 2], 2)],
                lambda x, w: functional.conv_transpose2d(x, w, None, 2, 2),
                transpose_arguments((2, 2), (2, 2), (0, 0), (1, 1), 1),
            ),
        ]
        samples["conv_transpose1d"] = [
            Sample(
                "strided",
                [plain(pool, dtype, [2, 3, 7]), plain(pool, dtype, [3, 2, 4], 2)],
                lambda x, w: functional.conv_transpose1d(x, w, None, 3, 1, 2),
                {
                    "stride": 3,
                    "padding": 1,
                    "output_padding": 2,
                    "dilation": 1,
                    "groups": 1,
                },
            ),
        ]
        samples["rfft"] = [
            Sample(
                f"length {n}",
                [plain(pool, dtype, [3, n])],
                lambda x: torch.view_as_real(torch.fft.rfft(x)),
                {},
            )
            for n in (8, 9, 1, 64, 1024, 1000)
        ]
        spectrum = plain(pool, dtype, [3, 5, 2])
        samples["irfft"] = [
            Sample(
                f"length {n}",
                [spectrum],
                lambda x, n=n: torch.fft.irfft(
                    torch.view_as_complex(x.contiguous()), n
                ),
                {"length": n},
            )
            for n in (8, 9, 4, 12, 1024)
        ] + [
            Sample(
                f"length {n} from {k} frequencies",
                [plain(pool, dtype, [2, k, 2], 3)],
                lambda x, n=n: torch.fft.irfft(
                    torch.view_as_complex(x.contiguous()), n
                ),
                {"length": n},
            )
            for n, k in ((1024, 513), (1000, 600))
        ]
        samples["interpolate"] = [
            Sample(
                f"{mode} {h}x{w}",
                [picture],
                lambda x, h=h, w=w, mode=mode: functional.interpolate(
                    x, size=(h, w), mode=mode
                ),
                {"size": [h, w], "mode": mode},
            )
            for mode in ("nearest", "bilinear")
            for h, w in ((10, 14), (3, 4), (7, 3))
        ]
        logits = plain(pool, dtype, [4, 5])
        classes = Input(
            pool.explicit(
                "cross entropy classes", torch.tensor([0, 4, 2, 2], dtype=torch.int64)
            )
        )
        class_ = Input(
            pool.explicit("cross entropy class", torch.tensor(3, dtype=torch.int64))
        )
        samples["cross_entropy"] = [
            Sample(
                reduction,
                [logits, classes],
                lambda x, t, reduction=reduction: functional.cross_entropy(
                    x, t, reduction=reduction
                ),
                {"reduction": reduction},
            )
            for reduction in ("mean", "sum", "none")
        ] + [
            Sample(
                "one sample",
                [plain(pool, dtype, [5]), class_],
                lambda x, t: functional.cross_entropy(x, t),
                {"reduction": "mean"},
            ),
        ]
        samples["cosine_similarity"] = [
            Sample(
                "rows",
                [wide, plain(pool, dtype, [3, 5], 2)],
                lambda x, y: functional.cosine_similarity(x, y, 1, 1e-8),
                {"dim": 1, "eps": 1e-8},
            ),
            Sample(
                "broadcast",
                [wide, plain(pool, dtype, [5], 3)],
                lambda x, y: functional.cosine_similarity(x, y, -1, 1e-8),
                {"dim": -1, "eps": 1e-8},
            ),
            Sample(
                "columns",
                [rank3, plain(pool, dtype, [2, 3, 4], 2)],
                lambda x, y: functional.cosine_similarity(x, y, 0, 1e-8),
                {"dim": 0, "eps": 1e-8},
            ),
        ]
    if dtype in FLOATS:
        positions = Input(
            pool.explicit(
                "scatter positions",
                torch.tensor([[2, 0, 1, 1], [0, 1, 2, 0]], dtype=torch.int64),
            )
        )
        column_positions = Input(
            pool.explicit(
                "scatter column positions",
                torch.tensor([[2, 0, 3], [1, 3, 0]], dtype=torch.int64),
            )
        )
        repeated = Input(
            pool.explicit(
                "scatter repeated positions",
                torch.tensor([[2, 2, 0], [1, 1, 1]], dtype=torch.int64),
            )
        )
        source = plain(pool, dtype, [2, 4], 2)
        samples["scatter"] = [
            Sample(
                "rows",
                [matrix, positions, source],
                lambda x, i, s: torch.scatter(x, 0, i, s),
                {"dim": 0},
            ),
            Sample(
                "columns",
                [matrix, column_positions, source],
                lambda x, i, s: torch.scatter(x, 1, i, s),
                {"dim": 1},
            ),
        ]
        samples["scatter_add"] = [
            Sample(
                "repeated columns",
                [matrix, repeated, source],
                lambda x, i, s: torch.scatter_add(x, 1, i, s),
                {"dim": 1},
            ),
            Sample(
                "rows",
                [matrix, positions, source],
                lambda x, i, s: torch.scatter_add(x, 0, i, s),
                {"dim": 0},
            ),
        ]
        values = Input(
            pool.explicit(
                "bincount values", torch.tensor([1, 3, 3, 0, 1, 1], dtype=torch.int64)
            )
        )
        samples["bincount"] = [
            Sample(
                "weighted",
                [values, plain(pool, dtype, [6], 3)],
                lambda x, w: torch.zeros(5, dtype=w.dtype).scatter_add(0, x, w),
                {"bins": 5},
            ),
        ]
        scales = Input(
            pool.explicit(
                "block scales",
                torch.tensor([[0.5, 2.0, 1.5], [4.0, 0.25, 3.0]], dtype=torch.float32),
            )
        )
        samples["dequantize_blocks"] = [
            Sample(
                "blocks",
                [plain(pool, dtype, [4, 6]), scales],
                lambda x, s: (
                    x.to(torch.float32)
                    * s.repeat_interleave(2, 0).repeat_interleave(2, 1)
                ),
                {"block_rows": 2, "block_columns": 2},
            ),
        ]
    if dtype == "I64":
        values = Input(
            pool.explicit(
                "bincount values", torch.tensor([1, 3, 3, 0, 1, 1], dtype=torch.int64)
            )
        )
        samples["bincount"] = [
            Sample(
                "counts",
                [values],
                lambda x: torch.bincount(x, minlength=5),
                {"bins": 5},
            ),
        ]
    if dtype == "F32":
        logits = Input(
            pool.explicit(
                "routing logits",
                ((torch.arange(24) * 7) % 24).to(torch.float32).reshape(3, 8) / 10 - 1,
            )
        )
        bias = Input(
            pool.explicit(
                "routing bias", ((torch.arange(8) * 3) % 8).to(torch.float32) / 20
            )
        )
        samples["grouped_topk_routing"] = [
            Sample(
                "deepseek",
                [logits, bias],
                lambda x, b: grouped_topk_routing_reference(x, b, 4, 2, 3, 2.5),
                {"groups": 4, "kept_groups": 2, "top_k": 3, "scale": 2.5},
            ),
        ]
    if dtype == "I64":
        indices = Input(
            pool.explicit(
                "one-hot indices",
                torch.tensor([[0, 3, 1], [2, 2, 0]], dtype=torch.int64),
            )
        )
        samples["one_hot"] = [
            Sample(
                "matrix", [indices], lambda x: functional.one_hot(x, 4), {"classes": 4}
            ),
        ]
    return samples


def operations(pool):
    """Yields every operation's name, the data type its cases are filed under, and its samples."""
    for name, (function, domain) in UNARY.items():
        for dtype in ALL:
            yield name, dtype, unary_samples(pool, dtype, function, domain)
    for name, (function, domains) in BINARY.items():
        for dtype in ALL:
            yield name, dtype, binary_samples(pool, dtype, function, domains)
    for name, function in SCALAR.items():
        for dtype in ALL:
            yield name, dtype, scalar_samples(pool, dtype, function)
    for name, function in REDUCTIONS.items():
        for dtype in ALL:
            yield name, dtype, reduction_samples(pool, dtype, function)
    for name, function in DIMENSION.items():
        for dtype in ALL:
            yield name, dtype, dimension_samples(pool, dtype, function)
    for dtype in ALL:
        yield "argsort", dtype, sort_samples(pool, dtype)
        yield (
            "topk",
            dtype,
            topk_samples(pool, dtype, lambda result: (result.values, result.indices)),
        )
        yield (
            "topk_values",
            dtype,
            topk_samples(pool, dtype, lambda result: result.values),
        )
        yield (
            "topk_indices",
            dtype,
            topk_samples(pool, dtype, lambda result: result.indices),
        )
        yield "clamp", dtype, clamp_samples(pool, dtype)
        yield "lerp", dtype, [sample for sample in lerp_samples(pool, dtype) if sample]
        yield "where", dtype, where_samples(pool, dtype)
        yield "matmul", dtype, matmul_samples(pool, dtype)
        for name, samples in shape_samples(pool, dtype).items():
            yield name, dtype, samples
        for name, samples in indexing_samples(pool, dtype).items():
            yield name, dtype, samples
        for target in ALL:
            yield "cast", dtype, cast_samples(pool, dtype, target)
        for name, samples in composite_samples(pool, dtype).items():
            yield name, dtype, [sample.composite() for sample in samples]
    for dtype in FLOATS + BYTE_FLOATS:
        yield "contraction", dtype, contraction_samples(pool, dtype)
        for name, samples in normalization_samples(pool, dtype).items():
            yield name, dtype, samples
        yield "conv2d", dtype, convolution_samples(pool, dtype)
        for name, samples in pooling_samples(pool, dtype).items():
            yield name, dtype, samples
        yield "scaled_dot_product_attention", dtype, attention_samples(pool, dtype)
        yield "paged_attention", dtype, paged_attention_samples(pool, dtype)
        yield "paged_mla_attention", dtype, paged_mla_attention_samples(pool, dtype)
        yield "grouped_matmul", dtype, grouped_matmul_samples(pool, dtype)
        for samples in (
            causal_conv1d_samples(pool, dtype),
            gated_delta_rule_samples(pool, dtype),
        ):
            for name, cases in dict(samples).items():
                yield name, dtype, cases
        yield "rotary_embedding", dtype, rotary_samples(pool, dtype)
    yield "scaled_mm", "F8E4M3", scaled_matmul_samples(pool, "F8E4M3")
    for name, samples in quantization_samples(pool).items():
        yield name, "F32" if name == "quantize" else "I8", samples


def encode_tensor(tensor):
    # A fresh contiguous copy, since `contiguous` and `clone` keep the strides of an empty view, which a byte view rejects.
    copy = torch.empty(tensor.shape, dtype=tensor.dtype)
    copy.copy_(tensor.detach())
    tensor = copy
    return {
        "dtype": NAMES[tensor.dtype],
        "shape": list(tensor.shape),
        "bytes": tensor.reshape(-1).view(torch.uint8).numpy().tobytes().hex(),
    }


def error_text(error):
    message = str(error).strip().splitlines()[0] if str(error).strip() else ""
    return f"{type(error).__name__}: {message[:160]}"


def generate():
    pool = Pool()
    cases = []
    names = set()
    for operation, dtype, samples in operations(pool):
        for sample in samples:
            label = sample.label
            if operation == "cast":
                label = f"to {sample.attributes['Cast']} {label}"
            name = f"{operation} {dtype} {label}"
            assert name not in names, f"duplicate case {name}"
            names.add(name)
            case = {"case": name, "operation": operation, "dtype": dtype}
            if sample.is_composite:
                case["composite"] = sample.attributes
            elif sample.attributes is not None:
                case["attributes"] = sample.attributes
            case["inputs"] = [operand.encode() for operand in sample.inputs]
            try:
                result = sample.reference(
                    *[operand.build(pool) for operand in sample.inputs]
                )
                results = list(result) if isinstance(result, tuple) else [result]
                case["outputs"] = [encode_tensor(output) for output in results]
            except (
                RuntimeError,
                TypeError,
                ValueError,
                IndexError,
                NotImplementedError,
            ) as error:
                case["error"] = error_text(error)
            cases.append(case)
    lines = [json.dumps({"torch": torch.__version__})]
    lines += [
        json.dumps({"input": key, **encode_tensor(tensor)})
        for key, tensor in pool.tensors.items()
    ]
    lines += [json.dumps(case) for case in cases]
    return "\n".join(lines) + "\n"


def promotions():
    """Returns `torch.result_type` for each pair of promotion categories, with each operand a scalar or a vector."""
    dtypes = [
        torch.bool,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ]
    records = []
    for lhs_dtype in dtypes:
        for rhs_dtype in dtypes:
            for lhs_rank, rhs_rank in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                lhs = torch.ones(() if lhs_rank == 0 else (1,), dtype=lhs_dtype)
                rhs = torch.ones(() if rhs_rank == 0 else (1,), dtype=rhs_dtype)
                records.append(
                    {
                        "lhs_dtype": str(lhs_dtype).removeprefix("torch."),
                        "lhs_rank": lhs_rank,
                        "rhs_dtype": str(rhs_dtype).removeprefix("torch."),
                        "rhs_rank": rhs_rank,
                        "result": str(torch.result_type(lhs, rhs)).removeprefix(
                            "torch."
                        ),
                    }
                )
    return json.dumps(records, indent=2) + "\n"


def tiny(output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases.jsonl").write_text(generate())
    (output / "promotions.json").write_text(promotions())
