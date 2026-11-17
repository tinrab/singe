# singe-loader

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-loader.svg)](https://crates.io/crates/singe-loader)
[![Documentation](https://docs.rs/singe-loader/badge.svg)](https://docs.rs/singe-loader)
![License](https://img.shields.io/crates/l/singe-loader.svg)

Loads the NVIDIA libraries behind Singe's `*-sys` crates at run time, so a program that uses them builds and starts on a machine without them.
A function whose library or symbol is missing returns a fixed value of its return type instead of failing to link.

This crate provides:

- **`Library`:** one library made of one or more files, with a `const` constructor per NVIDIA library, such as `Library::cublas()` or `Library::nvjpeg2k()`, and `Library::new` for any other.
- **`functions!`:** turns the declarations of a bindgen `extern "C"` block into functions of the same names and signatures that look up their symbol on first call.
- **`Missing`:** what a function returns when it cannot be called: zero for integers, null for pointers, and nothing for `()`, with `missing_status!` and `missing_value!` to implement it for a library's own types.
- **`take_absence`:** returns and clears the `Absence` that the last status-returning call on the current thread recorded, so a safe wrapper can turn that status into a load error.
- **`LoadError`:** a library with no files on this platform, a library that would not load, with each `Attempt` the loader made, or a missing symbol.

## Example

Load cuBLAS and print where each of its files was found, or each place the loader tried:

```rust,no_run
use singe_loader::{Library, LoadError};

static CUBLAS: Library = Library::cublas();

fn main() -> anyhow::Result<()> {
    match CUBLAS.paths() {
        Ok(paths) => {
            for path in paths {
                println!("{} loaded from {}", CUBLAS.name(), path.display());
            }
        }
        Err(LoadError::NotFound { library, attempts }) => {
            for attempt in &attempts {
                println!("{library}: {}: {}", attempt.path.display(), attempt.reason);
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}
```

A library loads at most once and stays loaded for the life of the process, so the functions that cached its symbols stay valid.
Its load result is cached too: a library that failed to load returns the same `LoadError` on every later call.

## Search order

The loader tries each file by name through the system loader's search first, then in the library directories under `CUDA_PATH` and `CUDA_HOME`, and on Linux under `/usr/local/cuda` and in `/run/opengl-driver/lib`.
The NVIDIA libraries have file names only on Linux and Windows; elsewhere, loading one returns `LoadError::Unsupported`.
