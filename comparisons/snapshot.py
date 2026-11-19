"""Prints the snapshot directory of a model of `cases.toml`, found as the verifications find theirs.

`PYTHONPATH=singe-model/verifications uv run python comparisons/snapshot.py MODEL`, from the repository root.
"""

import sys
import tomllib
from pathlib import Path

import support

CASES = Path(__file__).resolve().parent / "cases.toml"


def main():
    (model,) = sys.argv[1:]
    config = tomllib.loads(CASES.read_text())
    name = next(
        (entry["snapshot"] for entry in config["models"] if entry["name"] == model),
        None,
    )
    if name is None:
        sys.exit(f"{CASES} names no model {model}")
    directory = support.find_snapshot(name)
    if directory is None:
        sys.exit(
            f"snapshot {name} is neither in the Hugging Face cache nor set in SINGE_SNAPSHOT_{name}"
        )
    print(directory)


if __name__ == "__main__":
    main()
