"""Writing, reading, and comparing references."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
TENSORS = "tensors.safetensors"
METADATA = "reference.json"


def write(directory, tensors, metadata):
    """Writes a reference: `tensors`, by name, and `metadata`, whose keys come out sorted so a rewrite changes nothing."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if tensors:
        save_file(
            {
                name: tensor.detach().contiguous().cpu()
                for name, tensor in tensors.items()
            },
            directory / TENSORS,
        )
    (directory / METADATA).write_text(
        json.dumps(metadata, indent=1, sort_keys=True) + "\n"
    )


def read(directory):
    directory = Path(directory)
    tensors = load_file(directory / TENSORS) if (directory / TENSORS).exists() else {}
    return tensors, json.loads((directory / METADATA).read_text())


def half_gap(half, wide):
    """How far a half-precision result is from the same weights' `F32` one, worst and on average, which tolerances can be stated against."""
    difference = (half.float() - wide.float()).abs()
    return {"worst": float(difference.max()), "mean": float(difference.mean())}


def unaligned(text):
    """Markdown with each table cell's padding removed, since the repository's formatter re-pads tables."""
    rows = []
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.split("|")]
            line = "|".join(
                "-" if cell and set(cell) <= {"-", ":"} else cell for cell in cells
            )
        rows.append(line)
    return "\n".join(rows)


def same(left, right):
    """Whether two files hold the same values: tensors exactly, JSON by value, Markdown without table padding, anything else by bytes."""
    left, right = Path(left), Path(right)
    if not right.exists():
        return False
    if left.suffix == ".safetensors":
        first, second = load_file(left), load_file(right)
        return first.keys() == second.keys() and all(
            first[key].dtype == second[key].dtype
            and torch.equal(first[key], second[key])
            for key in first
        )
    if left.suffix == ".json":
        return json.loads(left.read_text()) == json.loads(right.read_text())
    if left.suffix == ".md":
        return unaligned(left.read_text()) == unaligned(right.read_text())
    return left.read_bytes() == right.read_bytes()


def differences(fresh, existing):
    """How `fresh`, a file or a tree, differs from `existing`, each file by value."""
    fresh, existing = Path(fresh), Path(existing)
    if fresh.is_file():
        return [] if same(fresh, existing) else [f"{existing} differs"]
    return [
        f"{existing / name} differs"
        for name in sorted(
            path.relative_to(fresh) for path in fresh.rglob("*") if path.is_file()
        )
        if not same(fresh / name, existing / name)
    ]


class Tensors:
    """The tensors a suite names as it builds its metadata: calling it with a tensor keeps the tensor and returns `{"tensor": name}` for the metadata to hold in its place."""

    def __init__(self):
        self.named = {}

    def __call__(self, tensor):
        name = f"t{len(self.named)}"
        self.named[name] = tensor.detach().cpu().contiguous().clone()
        return {"tensor": name}
