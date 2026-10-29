"""Check the selected matrix PTX against a host reference on a CUDA GPU.

Run with uv run --no-project python <this-file>. Requires libcuda and a GPU
supporting sm_80 instructions. The CPU emulator's tests do not use this script.
"""

import argparse
import ctypes as c
import json
import math
from pathlib import Path
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emulator-results",
        type=Path,
        help="JSON exported by the matrix integration test",
    )
    options = parser.parse_args()
    comparisons = {}
    if options.emulator_results:
        for case in json.loads(options.emulator_results.read_text()):
            comparisons.setdefault((case["m"], case["n"], case["k"]), []).append(case)
    cuda = c.CDLL("libcuda.so.1")
    pointer = c.c_void_p
    device_pointer = c.c_uint64
    uint = c.c_uint
    size = c.c_size_t
    signatures = {
        "cuInit": [uint],
        "cuDeviceGet": [c.POINTER(c.c_int), c.c_int],
        "cuCtxCreate_v2": [c.POINTER(pointer), uint, c.c_int],
        "cuCtxDestroy_v2": [pointer],
        "cuModuleLoadData": [c.POINTER(pointer), pointer],
        "cuModuleUnload": [pointer],
        "cuModuleGetFunction": [c.POINTER(pointer), pointer, c.c_char_p],
        "cuMemAlloc_v2": [c.POINTER(device_pointer), size],
        "cuMemFree_v2": [device_pointer],
        "cuMemcpyHtoD_v2": [device_pointer, pointer, size],
        "cuMemcpyDtoH_v2": [pointer, device_pointer, size],
        "cuLaunchKernel": [
            pointer,
            uint,
            uint,
            uint,
            uint,
            uint,
            uint,
            uint,
            pointer,
            c.POINTER(pointer),
            pointer,
        ],
        "cuCtxSynchronize": [],
    }
    for name, arguments in signatures.items():
        function = getattr(cuda, name)
        function.argtypes = arguments
        function.restype = c.c_int

    def call(name, *arguments):
        status = getattr(cuda, name)(*arguments)
        if status != 0:
            raise RuntimeError(f"{name} failed with CUDA status {status}")

    call("cuInit", 0)
    device = c.c_int()
    call("cuDeviceGet", c.byref(device), 0)
    context = pointer()
    call("cuCtxCreate_v2", c.byref(context), 0, device)
    module = pointer()
    try:
        source = Path(__file__).with_name("matrix_m16n8k16.ptx").read_bytes()
        call("cuModuleLoadData", c.byref(module), c.c_char_p(source))
        kernel = pointer()
        call("cuModuleGetFunction", c.byref(kernel), module, b"matrix")
        for m, n, k in [
            (0, 9, 17),
            (17, 0, 17),
            (17, 9, 0),
            (1, 1, 1),
            (16, 8, 16),
            (17, 9, 17),
            (33, 17, 33),
        ]:
            # Small binary fractions keep every product and accumulated sum
            # exact in FP32, independently of the hardware's reduction order.
            a = [((i * 17 + 3) % 31 - 15) / 8 for i in range(m * k)]
            b = [((i * 13 + 7) % 29 - 14) / 8 for i in range(k * n)]
            expected = [
                math.fsum(a[row * k + inner] * b[inner * n + col] for inner in range(k))
                for row in range(m)
                for col in range(n)
            ]
            allocations = []
            try:
                arguments = []
                for data in [
                    struct.pack(f"<{len(a)}e", *a),
                    struct.pack(f"<{len(b)}e", *b),
                    bytes(m * n * 4),
                ]:
                    address = device_pointer()
                    call("cuMemAlloc_v2", c.byref(address), max(len(data), 1))
                    allocations.append(address)
                    if data:
                        host = c.create_string_buffer(data)
                        call("cuMemcpyHtoD_v2", address, host, len(data))
                    arguments.append(address)
                arguments.extend([uint(m), uint(n), uint(k)])
                parameters = (pointer * len(arguments))(
                    *(c.cast(c.byref(argument), pointer) for argument in arguments)
                )
                call(
                    "cuLaunchKernel",
                    kernel,
                    max((n + 7) // 8, 1),
                    max((m + 15) // 16, 1),
                    1,
                    32,
                    1,
                    1,
                    0,
                    None,
                    parameters,
                    None,
                )
                call("cuCtxSynchronize")
                host = c.create_string_buffer(m * n * 4)
                if m * n:
                    call("cuMemcpyDtoH_v2", host, allocations[2], m * n * 4)
                actual = struct.unpack(f"<{m * n}f", host.raw)
                if tuple(expected) != actual:
                    for index, (reference, value) in enumerate(zip(expected, actual)):
                        if reference != value:
                            raise AssertionError(
                                f"{m}x{n}x{k}, output {index}: {value} != {reference}"
                            )
                if options.emulator_results:
                    cases = comparisons.get((m, n, k), [])
                    if {case["asynchronous"] for case in cases} != {False, True}:
                        raise AssertionError(f"{m}x{n}x{k}: missing CPU waiting style")
                    for case in cases:
                        if tuple(case["values"]) != actual:
                            raise AssertionError(
                                f"{m}x{n}x{k}: GPU differs from CPU, "
                                f"asynchronous={case['asynchronous']}"
                            )
                compared = (
                    "host and both CPU waiting styles"
                    if options.emulator_results
                    else "host"
                )
                print(f"{m}x{n}x{k}: {len(actual)} outputs match {compared} exactly")
            finally:
                for address in reversed(allocations):
                    call("cuMemFree_v2", address)
    finally:
        if module:
            call("cuModuleUnload", module)
        call("cuCtxDestroy_v2", context)


if __name__ == "__main__":
    main()
