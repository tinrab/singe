# singe-cudss

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cudss.svg)](https://crates.io/crates/singe-cudss)
[![Documentation](https://docs.rs/singe-cudss/badge.svg)](https://docs.rs/singe-cudss)
![License](https://img.shields.io/crates/l/singe-cudss.svg)

Safe Rust wrappers for NVIDIA cuDSS, the CUDA direct sparse solver library.

This crate provides:

- Sparse direct solves with separate analysis, factorization, and solve phases.
- Real, complex, and extended-precision inputs.
- Batched systems, Schur complements, and solver configuration.
- Hybrid CPU/GPU execution, multi-GPU modes, and custom allocators.

## Example

Solve a sparse linear system in three phases: analysis, factorization, and solve.
Assume `a` is a CSR matrix, `b` is the right-hand side, and `x` will hold the solution:

```rust,ignore
use singe_cuda::stream::StreamBinding;
use singe_cudss::{config::Config, types::Phase};

context.set_stream(&stream)?;
let config = Config::create()?;
let mut data = context.create_data()?;
StreamBinding::Borrowed(stream.to_borrowed()).sync_scope(|| -> singe_cudss::error::Result<()> {
	for phase in [Phase::ANALYSIS, Phase::FACTORIZATION, Phase::SOLVE] {
		// SAFETY: descriptors match their buffers, sparse indices are valid,
		// and all solver resources survive completion on the selected stream.
		unsafe {
			context.execute(phase, &config, &mut data, &a, &mut x, &b)?;
		}
	}
	Ok(())
})?;
assert_eq!(data.info()?, 0);
```

`Config` holds the solver settings, and the data object carries state from one phase to the next.
Check `data.info()` before reading the solution.
The [complete example](examples/simple.rs) builds a 5x5 symmetric positive-definite system and checks its solution.

## More examples

- [Complex solve](examples/simple_complex.rs): solve a complex-valued system.
- [Batch solve](examples/simple_batch.rs): process multiple systems.
- [Schur complement](examples/simple_schur_complement.rs): extract a reduced system.
- [Hybrid execution](examples/simple_hybrid_execute_mode.rs): configure CPU/GPU execution.
- [Custom allocator](examples/simple_memory_handler.rs): install allocation callbacks.

See the [API reference](https://docs.rs/singe-cudss).
