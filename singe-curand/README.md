# singe-curand

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-curand.svg)](https://crates.io/crates/singe-curand)
[![Documentation](https://docs.rs/singe-curand/badge.svg)](https://docs.rs/singe-curand)
![License](https://img.shields.io/crates/l/singe-curand.svg)

Safe Rust wrappers for the NVIDIA cuRAND, for random number generation.
A `Generator` fills device memory on a CUDA stream, and a `HostGenerator` fills host slices on the calling thread with the same algorithms.

This crate provides:

- **Generators:** the `XORWOW`, `MRG32k3a`, `MTGP32`, `MT19937`, and Philox pseudorandom generators, and scrambled and unscrambled Sobol quasirandom generators, with seed, offset, ordering, and dimension settings.
- **Distributions:** integer, uniform, normal, log-normal, and Poisson output in `u32`, `u64`, `f32`, and `f64`, through the `generate_*` methods.
- **Device-side support:** `DiscreteDistribution`, a Poisson histogram for kernels that use cuRAND's device API, and the Sobol direction vectors and scramble constants in `quasi`.

## Example

Generate 1024 reproducible uniform values on the GPU and copy them to the host:

```rust,no_run
use singe_cuda::{context::Context, memory::DeviceMemory, stream::StreamOptions};
use singe_curand::{generator::Generator, types::RngType};

fn main() -> anyhow::Result<()> {
    let context = Context::create()?;
    let stream = context.create_stream(StreamOptions::default())?;

    let mut generator = Generator::create(RngType::PseudoPhilox4_32_10, &stream)?;
    generator.set_seed(1234)?;

    let mut output = DeviceMemory::<f32>::create_uninit(&context, 1024)?;
    let values = generator.generate_uniform_f32(&mut output)?.copy_to_host_vec()?;
    assert!(values.iter().all(|&value| value > 0.0 && value <= 1.0));
    Ok(())
}
```

`generate_uniform_f32` returns once the values are ready, as a view of the now-initialized output.
The `unsafe` `generate_*_async` methods enqueue generation on the generator's stream without waiting.

## More examples

- [Host API](./examples/host_api.rs): generate uniform values on the GPU and on the CPU from the same seed.
