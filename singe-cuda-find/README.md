singe-cuda-find
===

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->
[![Latest version](https://img.shields.io/crates/v/singe-cuda-find.svg)](https://crates.io/crates/singe-cuda-find)
[![Documentation](https://docs.rs/singe-cuda-find/badge.svg)](https://docs.rs/singe-cuda-find)
![License](https://img.shields.io/crates/l/singe-cuda-find.svg)

CUDA Toolkit and NVIDIA library discovery helpers for Singe build scripts and wrapper crates.

This crate centralizes the path probing used by the Singe CUDA ecosystem.
It looks for headers, libraries, and tools through explicit environment variables,
the active `PATH` or dynamic library path, and common CUDA installation roots.
The result is a small [`Dependency`](src/lib.rs) value containing include and library directories plus a detected version when the header exposes one.

## Discovery

Toolkit components prefer component-specific roots such as `CUBLAS_PATH` or
`CUSOLVER_ROOT`, then fall back to `CUDA_PATH` or `CUDA_HOME`, then common
platform locations.
Driver-owned libraries such as CUDA Driver and NVML are searched in driver library paths because they are not necessarily installed inside the CUDA Toolkit.

## Examples

```rust
fn main() -> singe_cuda_find::Result<()> {
    let cuda = singe_cuda_find::find_cuda_runtime()?
        .expect("CUDA runtime headers and library were not found");

    println!("cargo:include={}", cuda.include_path.display());
    println!("cargo:rustc-link-search=native={}", cuda.library_path.display());

    if let Some(version) = cuda.version {
        println!("cargo:warning=found CUDA runtime {version}");
    }

    Ok(())
}
```
