# singe-cuda-find

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cuda-find.svg)](https://crates.io/crates/singe-cuda-find)
[![Documentation](https://docs.rs/singe-cuda-find/badge.svg)](https://docs.rs/singe-cuda-find)
![License](https://img.shields.io/crates/l/singe-cuda-find.svg)

Finds the headers, libraries, and tools of the CUDA Toolkit and the NVIDIA libraries, for build scripts that compile against them.
The `*-sys` crates load their libraries at run time through `singe-loader` and do not need this crate.

This crate provides:

- **`find_*` functions:** one per component, from the CUDA runtime, NVRTC, and nvJitLink to cuBLAS, cuDNN, NCCL, nvCOMP, and nvTIFF, each returning `Result<Option<Dependency>>`.
- **`Dependency`:** the include and library directories that were found, and the version the header reports, such as `"13.0.0"`.
- **Driver and tool paths:** `find_cuda_driver_path`, `find_nvml`, and `find_nvcc`.
- **Cargo directives:** `emit_dependency_metadata`, `emit_library_path_metadata`, `emit_link_libraries`, and `emit_static_link_libraries` print the directives for what was found, and `emit_rerun_if_env_changed` prints one for every environment variable the search reads.

## Example

A build script that finds the CUDA runtime and prints the Cargo directives for it:

```rust,no_run
fn main() -> anyhow::Result<()> {
    singe_cuda_find::emit_rerun_if_env_changed();
    let cuda = singe_cuda_find::find_cuda_runtime()?.ok_or_else(|| {
        anyhow::anyhow!("the CUDA runtime headers and library were not found")
    })?;
    println!("cargo:include={}", cuda.include_path.display());
    println!(
        "cargo:rustc-link-search=native={}",
        cuda.library_path.display()
    );
    if let Some(version) = &cuda.version {
        println!("cargo:warning=found CUDA runtime {version}");
    }
    Ok(())
}
```

A `find_*` function returns `Ok(None)` when the component is not installed, and an error only when the search itself fails, such as on an invalid glob pattern.
The caller chooses which libraries to link.

## Search order

A toolkit component checks its own root variables first, such as `CUBLAS_PATH` or `CUSOLVER_ROOT`, then `CUDA_PATH` or `CUDA_HOME`, then common installation locations, `PATH`, and the dynamic library search path.
The CUDA driver and NVML are searched in the driver's library paths, since they are often installed outside the toolkit.
