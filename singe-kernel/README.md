# singe-kernel

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-kernel.svg)](https://crates.io/crates/singe-kernel)
[![Documentation](https://docs.rs/singe-kernel/badge.svg)](https://docs.rs/singe-kernel)
![License](https://img.shields.io/crates/l/singe-kernel.svg)

Reusable CPU and GPU kernels.

CPU kernels operate on host slices.
CUDA kernels use device buffers and streams from `singe-cuda`.
Kernel families include matmul, attention, normalization, convolution, audio, quantization, and fused inference operations.

The goal of this crate is to offer reusable kernels built on top of various backends, which currently contain CPU and CUDA/cuTile-rs.

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

The CUDA kernels are written with cuTile and work with buffers and streams from `singe-cuda`.
Enable the kernels you need through Cargo features.
For matrix multiplication:

```toml
singe-cuda = "*"
singe-kernel = { version = "*", features = ["kernel_matmul"] }
```

Here is the same matrix multiplication on the GPU:

```rust,no_run
use singe_cuda::{context::Context, device::Device, memory::DeviceMemory};
use singe_kernel::cuda::matmul::{self, MatmulConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let context = Context::retain_primary_for_device(Device::new(0))?;
	let stream = context.create_stream()?;
	let a = DeviceMemory::from_slice(&context, &[1.0_f32, 2.0, 3.0, 4.0])?;
	let b = DeviceMemory::from_slice(&context, &[5.0_f32, 6.0, 7.0, 8.0])?;
	let mut output = DeviceMemory::<f32>::zeroed(&context, 4)?;
	let config = MatmulConfig::row_major(2, 2, 2);

	stream.sync_scope(|_| {
		// SAFETY: the buffers match the 2×2 layouts and stay alive until
		// the scope finishes. Output is disjoint and exclusively written.
		Ok(unsafe { matmul::matmul_f32(&stream, &mut output, &a, &b, config) })
	})??;
	assert_eq!(output.copy_to_host_vec()?, [19.0, 22.0, 43.0, 50.0]);
	Ok(())
}
```

`MatmulConfig` sets the matrix dimensions and layout.
The stream scope waits for the kernel to finish before we copy the result back.

`f32` kernels are enabled by default.
Add features such as `dtype_f16` or `dtype_bf16` for other datatypes, or `full` for all kernels and datatypes.

See the [CUDA guide](docs/kernels.md) for the kernel table, launch details, and autotuning, or the [API reference](https://docs.rs/singe-kernel) for individual functions.
