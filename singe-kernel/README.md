singe-kernel
===

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->
[![Latest version](https://img.shields.io/crates/v/singe-kernel.svg)](https://crates.io/crates/singe-kernel)
[![Documentation](https://docs.rs/singe-kernel/badge.svg)](https://docs.rs/singe-kernel)
![License](https://img.shields.io/crates/l/singe-kernel.svg)

Reusable CPU and GPU kernels.


## Kernels

Kernels are grouped by operation family.

| Category | Modules | Purpose |
| --- | --- | --- |
| Activation | `cuda::activation` | Elementwise nonlinearities and gated activations. |
| Attention | `cpu::attention`, `cuda::attention` | Prefill/decode attention kernels and related attention variants. |
| Audio | `audio`, `cpu::audio`, `cuda::audio` | Audio frontend configuration and audio feature extraction helpers used by speech models. |
| Cast | `cuda::cast` | Device-to-device dtype conversion. |
| Convolution | `cpu::conv`, `cuda::conv` | Small convolution and layout-specific convolution helpers. |
| Creation | `cuda::creation` | Fill, zeros, ones, arange, and other device initialization kernels. |
| Elementwise | `cuda::elementwise` | Binary/ternary elementwise operations, clamps, masks, and selects. |
| Embedding | `cuda::embedding` | Token embedding table lookups. |
| FFT | `cpu::fft`, `cuda::fft` | Small FFT helpers, interleaved complex transforms, and normalization. |
| Fused | `cpu::fused`, `cuda::fused` | Inference-oriented fused operations that combine common model steps. |
| Gather | `cuda::gather` | Index-based gathers and row selection. |
| Mask | `cuda::mask` | Boolean masks, causal masks, and mask composition. |
| Matmul | `cpu::matmul`, `cuda::matmul` | GEMM, batched GEMM, ragged batched GEMM, and matvec variants. |
| MoE | `cpu::moe`, `cuda::moe` | Mixture-of-experts routing and alignment helpers. |
| Normalization | `cpu::normalization`, `cuda::normalization` | RMS norm, layer norm, group norm, sparsemax, and related reductions. |
| Pooling | `cuda::pooling` | Pooling operations over device tensors. |
| Positional | `cpu::positional`, `cuda::positional` | RoPE and positional embedding variants. |
| Quantization | `cpu::quantization`, `cuda::quantization` | Quantize, dequantize, int4 unpacking, fp8 helpers, and quantized matmul. |
| Recurrent | `cpu::recurrent`, `cuda::recurrent` | Recurrent/state-space preprocessing helpers. |
| Reduction | `cuda::reduction` | Sum, max, norm, and row/axis reductions. |
| Scalar | `cuda::scalar` | Scalar arithmetic and scalar comparisons on device slices. |
| Shape | `cpu::shape`, `cuda::shape` | Copies, transposes, padding, slicing, tiling, and layout transforms. |
| Softmax | `cuda::softmax` | Softmax and log-sum-exp style kernels. |
| Unary | `cuda::unary` | Single-input math operations and predicates. |


## Usage

CPU kernels operate on slices and return host-owned vecs:

```rust
use singe_kernel::cpu::matmul;

let lhs = [1.0_f32, 2.0, 3.0, 4.0];
let rhs = [5.0_f32, 6.0, 7.0, 8.0];

let out = matmul::matmul_f32(
    &lhs,
    &rhs,
    2,     // rows
    2,     // columns
    2,     // reduction
    2,     // lhs row stride
    2,     // rhs row stride
    false, // transpose lhs
    false, // transpose rhs
);

assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
```

CUDA kernels use `singe_cuda` streams and device memory:

```rust,no_run
use singe_cuda::{context::Context, memory::DeviceMemory};
use singe_kernel::cuda::{creation, scalar};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = Context::create()?;
    let stream = ctx.create_stream()?;

    let mut input = DeviceMemory::<f32>::create(4)?;
    let mut output = DeviceMemory::<f32>::create(4)?;

    creation::fill_f32(&stream, &mut input, 2.0)?;
    scalar::scale_f32(&stream, &mut output, &input, 3.0)?;
    stream.synchronize()?;

    assert_eq!(output.copy_to_host_vec()?, vec![6.0; 4]);

    Ok(())
}
```

These CUDA kernels are currently implemented using [cutile](https://crates.io/crates/cutile).
The public modules are intended to stay backend-oriented rather than cuTile-specific, so the crate can add other implementations later.
You can access cuTile kernels directly for more control.

To make it compile faster, we expose many feature flags that you need to enable based on which kernels you need.

```toml
singe-cuda = "*"
singe-kernel = { version = "*", features = [
    "cuda_13_3",
    "cutile",
    "dtype-f32",
    "kernel-creation",
    "kernel-scalar",
] }
```

Choose features based on the modules and data types you import.
The default feature set is intentionally small:

```toml
default = ["dtype-f32"]
```

CPU APIs are available without CUDA features.
CUDA APIs are opt-in:

- `cuda_13_3` and `cutile` make `singe_kernel::cuda` available.
- `kernel-*` flags expose CUDA operation families. For example, importing
  `singe_kernel::cuda::matmul` requires `kernel-matmul`.
- Dtype flags expose dtype-specific functions. For example, `scale_f32` requires
  `dtype-f32`, and `zeros_i32` requires `dtype-i32`.
- `kernels-all` enables every CUDA operation family.
- `full` enables `cutile`, `kernels-all`, and `dtypes-all`.

Data type support is also feature-gated:

- Floating point: `dtype-f32`, `dtype-f16`, `dtype-bf16`, `dtype-f64`,
  `dtype-f8`.
- Signed integers: `dtype-i8`, `dtype-i32`, `dtype-i64`.
- Unsigned integers: `dtype-u8`, `dtype-u32`, `dtype-u64`.
- `dtypes-all` enables every dtype flag.
