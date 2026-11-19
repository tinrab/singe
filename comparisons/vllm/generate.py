"""vLLM's side of the generation comparison: runs every case of `comparisons/cases.toml` for one model and prints one JSON line of timings per case, as Singe's `compare` example does.

`uv run python generate.py --model smollm2 --snapshot DIR [--case NAME]`.
Generation runs through vLLM's offline `LLM.generate`, in this process, with no server, compiled as vLLM runs in production: `torch.compile` and CUDA graphs.
"""

import time

# Startup counts from here, before the imports, as Singe's counts from its process start.
STARTED = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import tomllib  # noqa: E402
from pathlib import Path  # noqa: E402

CASES = Path(__file__).resolve().parent.parent / "cases.toml"
MASK = (1 << 64) - 1


def prompt_ids(config, row, length):
    """Row `row`'s prompt, as `cases.toml` defines it and `compare.rs` makes it."""
    state = row
    ids = []
    for _ in range(length):
        state = (state * 6364136223846793005 + 1442695040888963407) & MASK
        ids.append(config["first_id"] + state % config["id_span"])
    return ids


def find_libcuda():
    """Points Triton at `libcuda.so.1` on `LD_LIBRARY_PATH`, which it otherwise finds through `/sbin/ldconfig`, absent on some systems such as NixOS."""
    if os.environ.get("TRITON_LIBCUDA_PATH"):
        return
    for directory in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if directory and (Path(directory) / "libcuda.so.1").exists():
            os.environ["TRITON_LIBCUDA_PATH"] = directory
            return


def timed(llm, prompts, params, runs):
    """The median seconds of `runs` generations of `prompts`, after one untimed."""
    llm.generate(prompts, params, use_tqdm=False)
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        llm.generate(prompts, params, use_tqdm=False)
        times.append(time.perf_counter() - started)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--case", help="runs only the case of this name")
    arguments = parser.parse_args()

    config = tomllib.loads(arguments.cases.read_text())
    model = next(
        (model for model in config["models"] if model["name"] == arguments.model), None
    )
    if model is None:
        sys.exit(f"{arguments.cases} names no model {arguments.model}")

    find_libcuda()
    # vLLM's progress logging would bury the JSON lines the report reads.
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=arguments.snapshot,
        dtype="bfloat16",
        max_model_len=model["context"],
        gpu_memory_utilization=0.6,
        # Each case runs the same prompts several times, which vLLM's prefix cache would answer without prefilling them again; Singe has no prefix cache yet, so both prefill every run.
        enable_prefix_caching=False,
        seed=0,
    )
    startup = time.perf_counter() - STARTED

    new_tokens = config["new_tokens"]

    def sampling(tokens):
        return SamplingParams(
            temperature=0, max_tokens=tokens, min_tokens=tokens, ignore_eos=True
        )

    for case in config["cases"]:
        if arguments.case and case["name"] != arguments.case:
            continue

        prompts = [
            TokensPrompt(
                prompt_token_ids=prompt_ids(config, row, case["prompt_tokens"])
            )
            for row in range(case["batch"])
        ]
        # The first token is the prompt's prefill and one step; the later steps are what `new_tokens` adds to it.
        first = timed(llm, prompts, sampling(1), config["runs"])
        total = timed(llm, prompts, sampling(new_tokens), config["runs"])
        line = {
            "side": "vllm",
            "model": model["name"],
            "mode": "compiled",
            "case": case["name"],
            "batch": case["batch"],
            "prompt_tokens": case["prompt_tokens"],
            "new_tokens": new_tokens,
            "startup_s": {"engine": round(startup, 3)},
            "first_token_ms": round(first * 1e3, 3),
            "step_ms": round((total - first) / (new_tokens - 1) * 1e3, 4),
            "tokens_per_second": round(case["batch"] * new_tokens / total, 1),
        }
        print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
