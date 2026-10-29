# singe-cutensor

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cutensor.svg)](https://crates.io/crates/singe-cutensor)
[![Documentation](https://docs.rs/singe-cutensor/badge.svg)](https://docs.rs/singe-cutensor)
![License](https://img.shields.io/crates/l/singe-cutensor.svg)

Safe Rust wrappers for NVIDIA cuTENSOR, for tensor contractions and other tensor operations on the GPU.

This crate provides:

- Tensor descriptors with explicit dimensions, strides, and mode labels.
- Contractions, reductions, permutations, and elementwise operations.
- Reusable execution plans, workspace estimates, and plan caching.
- Multi-GPU and multi-process APIs through `mg` and `mp`.

## Example

Execute a prepared contraction as `D = alpha * contraction(A, B) + beta * C`.
Assume `contraction` describes the modes and layouts of the supplied device buffers:

```rust,ignore
use singe_cuda::memory::DeviceMemory;
use singe_cutensor::{plan::{Plan, PlanPreference}, types::WorkspacePreference};

let preference = PlanPreference::create_default(&context)?;
let size = Plan::estimate_workspace_size(
	&context, &contraction, &preference, WorkspacePreference::Default,
)?;
let plan = Plan::create(&context, &contraction, &preference, size)?;
let mut workspace = DeviceMemory::<u8>::create_uninit(
	&cuda, plan.required_workspace_size_bytes()?,
)?;
stream.sync_scope(|_| {
	// SAFETY: descriptors match these disjoint buffers, inputs are initialized,
	// and buffers, workspace, and plan remain alive through scope completion.
	Ok(unsafe {
		plan.contract(
			&1.0_f32, &a, &b, &0.0_f32, &c, &mut d,
			Some(&mut workspace), &stream,
		)
	})
})??;
```

## More examples

- [Einsum](examples/einsum.rs): express contractions and reductions with axis labels.
- [Reduction](examples/reduction.rs): reduce tensor dimensions.
- [Plan cache](examples/contraction_plan_cache.rs): reuse contraction planning work.
- [Multi-GPU contraction](examples/mg_contraction_multi_gpu.rs): distribute a contraction over GPUs.

See the [API reference](https://docs.rs/singe-cutensor).
