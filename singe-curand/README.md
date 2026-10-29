# singe-curand

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-curand.svg)](https://crates.io/crates/singe-curand)
[![Documentation](https://docs.rs/singe-curand/badge.svg)](https://docs.rs/singe-curand)
![License](https://img.shields.io/crates/l/singe-curand.svg)

Safe Rust wrappers for the NVIDIA cuRAND host API, for random number generation on the GPU or CPU.

This crate provides:

- GPU random-number generation through `Generator` and CPU generation through `HostGenerator`.
- Pseudorandom and quasirandom generators with seed and offset control.
- Integer, uniform, normal, log-normal, and Poisson distributions.
- Methods that wait for the generated values, and `unsafe` asynchronous methods that don't.

## Example

Generate reproducible uniform values on the GPU:

```rust
use std::sync::Arc;
use singe_cuda::{context::Context, memory::DeviceMemory};
use singe_curand::{generator::Generator, types::RngType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let context = Context::create()?;
	let stream = Arc::new(context.create_stream()?);
	let mut generator = Generator::create(RngType::PseudoPhilox4_32_10, &stream)?;
	generator.set_seed(1234)?;
	let mut output = DeviceMemory::<f32>::create_uninit(&context, 1024)?;
	let values = generator.generate_uniform_f32(&mut output)?.copy_to_host_vec()?;
	assert!(values.iter().all(|&value| value > 0.0 && value <= 1.0));
	Ok(())
}
```

The generator keeps its stream and context alive, and the method returns once the values are ready.
Use `HostGenerator` to fill host memory, or the `unsafe` `generate_*_async` methods to queue generation without waiting.

## More examples

- [Host API](examples/host_api.rs): generate on the GPU and CPU with a fixed seed.

See the [API reference](https://docs.rs/singe-curand).
