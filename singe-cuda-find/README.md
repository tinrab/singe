# singe-cuda-find

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cuda-find.svg)](https://crates.io/crates/singe-cuda-find)
[![Documentation](https://docs.rs/singe-cuda-find/badge.svg)](https://docs.rs/singe-cuda-find)
![License](https://img.shields.io/crates/l/singe-cuda-find.svg)

Discovery tools for the CUDA Toolkit and NVIDIA libraries.

This crate provides:

- Header, library, and tool discovery for CUDA and NVIDIA libraries.
- Per-library root variables such as `CUBLAS_PATH`, falling back to the CUDA installation.
- Separate discovery for driver libraries such as the CUDA driver and NVML.
- A `Dependency` value with the include path, library path, and header version found.

## Example

Find the CUDA runtime in a build script:

```rust
fn main() -> singe_cuda_find::Result<()> {
	let cuda = singe_cuda_find::find_cuda_runtime()?
		.expect("CUDA runtime headers and library were not found");
	println!("cargo:rustc-link-search=native={}", cuda.library_path.display());
	println!("cargo:rustc-link-lib=cudart");
	Ok(())
}
```

Discovery returns `None` if the dependency is not installed, and an error if the search itself fails.
The caller chooses which libraries to link.

## Configuration

Discovery checks the per-library variables first, then `CUDA_PATH` or `CUDA_HOME`, then common installation locations.
It also searches `PATH` and the dynamic library search path.
Driver libraries are searched separately, since they are often installed outside the toolkit.
