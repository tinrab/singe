"""Writes or checks references: `uv run --locked python -m references [SUITE...] [--check] [--real] [--prepare]`, from `scripts/`.

Without `--real`, each suite writes what it computes from tiny random models and fixed inputs, and `--check` compares a fresh run with what is on disk.
With `--real`, each suite writes what it computes from real snapshots in the Hugging Face cache, which a check cannot reproduce without them.
With `--prepare`, every suite whose output the tests read is written, unless the last prepare wrote it with the same scripts and `uv.lock`.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
STAMPS = ROOT / "target/references"
# These suites write checked-in sources rather than what the tests read.
CHECKED_IN = {"onnx_schemas"}


def digest(real):
    """A digest of everything that decides what a prepare writes."""
    hasher = hashlib.sha256(b"real" if real else b"tiny")
    for path in sorted(PACKAGE.rglob("*.py")) + [ROOT / "uv.lock"]:
        hasher.update(str(path.relative_to(ROOT)).encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def current(stamp, expected):
    """Whether `stamp` records a prepare with `expected` whose files all still exist."""
    try:
        recorded = json.loads(stamp.read_text())
    except OSError, ValueError:
        return False
    return recorded.get("digest") == expected and all(
        (ROOT / name).is_file() for name in recorded.get("files", [])
    )


def run_each(names, flags):
    """Runs each suite in a process of its own, since a suite can leave PyTorch's global state changed, such as its thread count, which moves another suite's half-precision rounding."""
    return [
        name
        for name in names
        if subprocess.run(
            [sys.executable, "-m", "references", name, *flags], check=False
        ).returncode
        != 0
    ]


def prepare(real):
    stamp = STAMPS / ("real.json" if real else "tests.json")
    expected = digest(real)
    if current(stamp, expected):
        print("references: current")
        return

    from .suites import SUITES

    names = [
        name
        for name, suite in SUITES.items()
        if name not in CHECKED_IN and hasattr(suite, "real" if real else "tiny")
    ]
    # Some file systems store modification times in whole seconds.
    started = time.time() - 1
    failed = run_each(names, ["--real"] if real else [])
    if failed:
        sys.exit(f"failed suites: {failed}")

    written = sorted(
        {
            str(path.relative_to(ROOT))
            for name in names
            for path in SUITES[name].OUTPUT.rglob("*")
            if path.is_file() and path.stat().st_mtime >= started
        }
    )
    STAMPS.mkdir(parents=True, exist_ok=True)
    stamp.write_text(
        json.dumps({"digest": expected, "files": written}, indent=1) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="*", help="suites to run; every one if none")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare with the references on disk instead of writing them",
    )
    parser.add_argument(
        "--real", action="store_true", help="write the references real snapshots give"
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="write what the tests read, unless it is current",
    )
    arguments = parser.parse_args()
    if arguments.prepare:
        if arguments.suites or arguments.check:
            parser.error("--prepare takes no suites and no --check")
        prepare(arguments.real)
        return

    from .format import differences
    from .suites import SUITES

    unknown = [name for name in arguments.suites if name not in SUITES]
    if unknown:
        parser.error(f"unknown suites {unknown}, of {', '.join(SUITES)}")
    names = arguments.suites or list(SUITES)
    if len(names) > 1:
        flags = [
            flag
            for flag, set_ in [("--check", arguments.check), ("--real", arguments.real)]
            if set_
        ]
        failed = run_each(names, flags)
        if failed:
            sys.exit(f"failed suites: {failed}")
        return

    failed = []
    for name in names:
        suite = SUITES[name]
        run = getattr(suite, "real" if arguments.real else "tiny", None)
        if run is None:
            continue
        if not arguments.check:
            run(suite.OUTPUT)
            print(f"{name}: written")
            continue
        with tempfile.TemporaryDirectory() as directory:
            fresh = Path(directory) / name
            run(fresh)
            # Only what this run writes is compared, so real references and recorded baselines may sit beside what it writes.
            found = [
                difference
                for written in sorted(
                    path.relative_to(fresh)
                    for path in fresh.rglob("*")
                    if path.is_file()
                )
                for difference in differences(fresh / written, suite.OUTPUT / written)
            ]
            shutil.rmtree(fresh, ignore_errors=True)
        if found:
            failed.append(name)
            print(f"{name}: {found}", file=sys.stderr)
        else:
            print(f"{name}: matches")
    if failed:
        sys.exit(f"stale references: {failed}")


main()
