# singe-cufft

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cufft.svg)](https://crates.io/crates/singe-cufft)
[![Documentation](https://docs.rs/singe-cufft/badge.svg)](https://docs.rs/singe-cufft)
![License](https://img.shields.io/crates/l/singe-cufft.svg)

Safe Rust wrappers for NVIDIA cuFFT, for Fourier transforms on the GPU.

This crate provides:

- One-, two-, and three-dimensional FFT plans and batched transforms.
- Complex-to-complex, real-to-complex, and complex-to-real execution.
- Stream binding and explicit workspace management.
- The cuFFTXt API, with more data types and transforms split across several GPUs.

## Example

Transform eight complex values in place, then apply the inverse transform.
Assume `context` and `stream` are an existing CUDA context and one of its streams:

```rust,ignore
use singe_cuda::{
	memory::DeviceMemory, stream::StreamBinding, types::Complex32,
	view::DeviceOutputView,
};
use singe_cufft::{plan::PlanBuilder, types::{Direction, TransformType}};

let mut values = DeviceMemory::from_slice(&context, &[Complex32::new(1.0, 0.0); 8])?;
let plan = PlanBuilder::new(&context)
	.one_dimensional(8, TransformType::ComplexToComplex)
	.create()?;
StreamBinding::Borrowed(stream.to_borrowed()).sync_scope(|| -> singe_cufft::error::Result<()> {
	// SAFETY: the plan covers all eight initialized elements, and the buffer
	// remains exclusively borrowed through completion of both transforms.
	unsafe {
		for direction in [Direction::Forward, Direction::Inverse] {
			plan.execute_c2c_f32_in_place_async(
				DeviceOutputView::from_output(&mut values), direction, Some(&stream),
			)?;
		}
	}
	Ok(())
})?;
let result = values.copy_to_host_vec()?;
```

cuFFT does not normalize the inverse transform, so this round trip multiplies each original value by eight.
The [complete example](examples/one_dimensional_c2c.rs) also demonstrates batching and verifies the output.

## More examples

- [Real/complex round trip](examples/one_dimensional_r2c_c2r.rs): transform real input and reconstruct it.
- [Two-dimensional transforms](examples/two_dimensional_c2r_r2c.rs): work with a 2-D real/complex layout.
- [Three-dimensional transforms](examples/three_dimensional_c2c.rs): transform complex volume data.

See the [API reference](https://docs.rs/singe-cufft).
