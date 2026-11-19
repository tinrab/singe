"""Prints the comparison's table from the JSON lines both sides wrote, one table per model and mode.

`uv run python report.py LINES...`, each file holding lines from `compare.rs` or `generate.py`.
Each row is a case; each measure shows Singe's value, vLLM's, and Singe's as a share of vLLM's, under 1 when Singe is faster.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def startup(line):
    """A side's seconds from process start to ready to generate: Singe's runtime and load, vLLM's engine construction."""
    phases = line["startup_s"]
    return phases.get("engine", phases.get("runtime", 0) + phases.get("load", 0))


def cell(singe, vllm, faster_is_lower=True):
    if singe is None or vllm is None:
        return f"{shown(singe):>9} {shown(vllm):>9} {'':>6}"
    ratio = singe / vllm if faster_is_lower else vllm / singe
    return f"{singe:9.2f} {vllm:9.2f} {ratio:6.2f}"


def shown(value):
    return "—" if value is None else f"{value:.2f}"


def measure(key, line):
    return line[key] if line else None


def main():
    lines = [
        json.loads(text)
        for path in sys.argv[1:]
        for text in Path(path).read_text().splitlines()
        if text.startswith("{")
    ]
    tables = defaultdict(dict)
    for line in lines:
        tables[(line["model"], line["mode"])].setdefault(line["case"], {})[
            line["side"]
        ] = line

    for (model, mode), cases in tables.items():
        sides = next(iter(cases.values()))
        singe, vllm = sides.get("singe"), sides.get("vllm")
        print(
            f"\n{model}, {mode}: startup {shown(singe and startup(singe))} s against {shown(vllm and startup(vllm))} s"
        )
        print(
            f"{'case':10} {'batch':>5} {'prompt':>6} | {'first token, ms':^26} | {'step, ms':^26} | {'tokens per second':^26}"
        )
        for name, sides in cases.items():
            singe, vllm = sides.get("singe"), sides.get("vllm")
            either = singe or vllm
            print(
                f"{name:10} {either['batch']:5} {either['prompt_tokens']:6} | "
                f"{cell(measure('first_token_ms', singe), measure('first_token_ms', vllm))} | "
                f"{cell(measure('step_ms', singe), measure('step_ms', vllm))} | "
                f"{cell(measure('tokens_per_second', singe), measure('tokens_per_second', vllm), faster_is_lower=False)}"
            )


if __name__ == "__main__":
    main()
