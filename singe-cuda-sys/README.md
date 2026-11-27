# singe-cuda-sys

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cuda-sys.svg)](https://crates.io/crates/singe-cuda-sys)
[![Documentation](https://docs.rs/singe-cuda-sys/badge.svg)](https://docs.rs/singe-cuda-sys)
![License](https://img.shields.io/crates/l/singe-cuda-sys.svg)

Low-level FFI bindings for the CUDA driver and runtime APIs, NVRTC, NVVM, nvJitLink, NVTX, and the CUDA profiler.

Most users want the safe crate, [`singe-cuda`](../singe-cuda/README.md).

## Bindings

| Binds                                   | Header               | Module          | Feature          |
| --------------------------------------- | -------------------- | --------------- | ---------------- |
| The CUDA driver API                     | `cuda.h`             | `driver`        | `driver_13_4`    |
| The CUDA runtime API                    | `cuda_runtime_api.h` | `runtime`       | `runtime_13_4`   |
| CUDA profiler start and stop            | `cudaProfiler.h`     | `profiler`      | `driver_13_4`    |
| Data types shared by the CUDA libraries | `library_types.h`    | `library_types` | `runtime_13_4`   |
| NVRTC, which compiles CUDA C++          | `nvrtc.h`            | `nvrtc`         | `nvrtc_13_4`     |
| NVVM, which compiles NVVM IR            | `nvvm.h`             | `nvvm`          | `nvvm_13_4`      |
| nvJitLink, which links device code      | `nvJitLink.h`        | `nvjitlink`     | `nvjitlink_13_4` |
| NVTX annotations                        | `nvToolsExt.h`       | `nvtx`          | `nvtx_3`         |

Every feature is on by default.
The bindings are generated with bindgen from the CUDA 13.4 and NVTX 3 headers.

## Loading

Nothing links at build time.
[`singe-loader`](../singe-loader/README.md) loads each library on the first call of one of its functions, or up front with `load_driver`, `load_runtime`, `load_nvrtc`, `load_nvvm`, `load_nvjitlink`, or `load_nvtx`.
When a library or one of its symbols is missing, a function that returns a status returns an error status, and `take_absence` reports what was missing.
A function without a status returns zero, null, or nothing instead.
`provide_driver` and `provide_runtime` serve the driver and runtime from an in-process implementation instead, such as [`singe-cuda-emulator`](../singe-cuda-emulator/README.md).
