# singe-cufft

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cufft.svg)](https://crates.io/crates/singe-cufft)
[![Documentation](https://docs.rs/singe-cufft/badge.svg)](https://docs.rs/singe-cufft)
![License](https://img.shields.io/crates/l/singe-cufft.svg)

Safe Rust wrappers for cuFFT, NVIDIA's library for fast Fourier transforms on the GPU.
A `Plan` describes a transform, manages its workspace, and runs it on a stream.

This crate provides:

- **Plans:** one-, two-, and three-dimensional transforms and batched strided layouts, through `plan_1d`, `plan_2d`, `plan_3d`, and `plan_many`.
- **Transforms:** complex-to-complex, real-to-complex, and complex-to-real, in single and double precision.
- **Workspace control:** cuFFT allocates the workspace during planning by default; `set_auto_allocation(false)` with `set_work_area` installs one you own.
- **cuFFTXt:** element types given as CUDA data types, link-time callbacks, and transforms split across several GPUs, in `xt`.
- **Device functions:** the experimental `device` module describes FFTs and extracts their LTO-IR or fatbinary code for linking into your own kernels with nvJitLink.

## Example

Transform eight complex values in place, apply the inverse, and check the result:

```rust,no_run
use singe_cuda::{context::Context, memory::DeviceMemory, stream::StreamOptions, types::Complex32};
use singe_cufft::{
    plan::Plan,
    types::{Direction, TransformType},
};

fn main() -> anyhow::Result<()> {
    let context = Context::create()?;
    let stream = context.create_stream(StreamOptions::default())?;

    let input = [Complex32::new(1.0, 0.0); 8];
    let mut values = DeviceMemory::from_slice(&context, &input)?;

    let mut plan = Plan::create(&context)?;
    plan.plan_1d(input.len(), TransformType::ComplexToComplex, 1)?;
    plan.execute_in_place(&mut values, Direction::Forward, &stream)?;
    plan.execute_in_place(&mut values, Direction::Inverse, &stream)?;

    // cuFFT does not normalize the inverse transform, so the round trip scales each value by eight.
    let output = values.copy_to_host_vec()?;
    assert!(output.iter().all(|value| (value - Complex32::new(8.0, 0.0)).norm() < 1.0e-3));
    Ok(())
}
```

`execute_in_place` returns once the transform has finished on `stream`.
When a plan runs on a different stream than last time, the new stream first waits for the work queued on the previous one.
`Plan::execute_async` and `execute_in_place_async` are `unsafe` and only enqueue the transform, and the `_scoped` forms enqueue it into a `StreamScope`.

## More examples

- [1-D complex round trip](./examples/one_dimensional_c2c.rs): a batched in-place complex-to-complex transform and its inverse.
- [1-D real round trip](./examples/one_dimensional_r2c_c2r.rs): a real-to-complex transform into the half spectrum, and the complex-to-real inverse.
- [2-D real round trip](./examples/two_dimensional_c2r_r2c.rs): batched 2-D complex-to-real and real-to-complex transforms planned with `plan_many`.
- [3-D complex round trip](./examples/three_dimensional_c2c.rs): an in-place 3-D complex transform planned with `plan_many`.
