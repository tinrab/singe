# singe-kernel

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-kernel.svg)](https://crates.io/crates/singe-kernel)
[![Documentation](https://docs.rs/singe-kernel/badge.svg)](https://docs.rs/singe-kernel)
![License](https://img.shields.io/crates/l/singe-kernel.svg)

Reusable CPU and GPU kernels for inference.
CPU kernels work on host slices, and CUDA kernels, written with cuTile, work on device buffers and streams from `singe-cuda`.

This crate provides:

- **CPU kernels** in `cpu`: matmul, attention, convolution, FFT, normalization, positional encodings, quantization, mixture-of-experts layers, and gated delta rule preprocessing.
- **CUDA kernels** in `cuda`, one Cargo feature per family, from activations and attention to softmax and quantized matmul.
- **Autotuning** for the CUDA kernels with tunable configurations: tune on a cache miss, require a cached result, or launch with an explicit tuning value.

## Example

Multiply two row-major matrices on the CPU:

```rust
use singe_kernel::cpu::matmul;

fn main() {
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [5.0_f32, 6.0, 7.0, 8.0];
    let output = matmul::matmul_f32(
        &a, &b,
        2, 2, 2,      // rows, columns, reduction
        2, 2,         // input row strides
        false, false, // transpose inputs
    );
    assert_eq!(output, [19.0, 22.0, 43.0, 50.0]);
}
```

## CUDA kernels

Enable the kernel families you use through Cargo features:

```toml
singe-cuda = "0.2"
singe-kernel = { version = "0.2", features = ["kernel_matmul"] }
```

The same matrix multiplication on the GPU:

```rust,ignore
use singe_cuda::{context::Context, memory::DeviceMemory, stream::StreamOptions};
use singe_kernel::cuda::matmul::{self, MatmulConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::create()?;
    let stream = context.create_stream(StreamOptions::default())?;
    let a = DeviceMemory::from_slice(&context, &[1.0_f32, 2.0, 3.0, 4.0])?;
    let b = DeviceMemory::from_slice(&context, &[5.0_f32, 6.0, 7.0, 8.0])?;
    let mut output = DeviceMemory::<f32>::zeroed(&context, 4)?;

    // SAFETY: the buffers match the 2×2 layouts, do not overlap,
    // and stay alive until `stream.synchronize` returns.
    unsafe { matmul::matmul_f32(&stream, &mut output, &a, &b, MatmulConfig::row_major(2, 2, 2))? };
    stream.synchronize()?;
    assert_eq!(output.copy_to_host_vec()?, [19.0, 22.0, 43.0, 50.0]);
    Ok(())
}
```

`MatmulConfig` sets the matrix dimensions, row strides, and transposes, and its `tuning` field chooses between the default launch and a tuned one.
The [CUDA guide](./docs/kernels.md) lists every kernel family and describes launches and autotuning.

## Features

- `kernel_<family>`, such as `kernel_matmul` or `kernel_attention`: one CUDA kernel family; `kernels_all` enables every family.
- `dtype_<type>`, such as `dtype_f16` or `dtype_bf16`: kernels for one element type; `dtype_f32` is on by default, and `dtypes_all` enables every type.
- `full`: every kernel family and element type.
- `serde`: `Serialize` and `Deserialize` for kernel configurations and tuning reports, so a tuning result can be saved and reused.
