# singe-cudnn

[![Latest version](https://img.shields.io/crates/v/singe-cudnn.svg)](https://crates.io/crates/singe-cudnn)
[![Documentation](https://docs.rs/singe-cudnn/badge.svg)](https://docs.rs/singe-cudnn)
![License](https://img.shields.io/crates/l/singe-cudnn.svg)

Rust wrappers for NVIDIA cuDNN, the deep neural network library, including a Rust port of NVIDIA's cuDNN Frontend.

This crate provides:

- Native cuDNN contexts, tensor descriptors, and backend operations.
- Frontend graphs for matmul, convolution, attention, normalization, and other operations.
- Plan candidates, filtering, compilation, autotuning, and serialization.
- Explicit storage bindings, owned execution, and CUDA graph capture and replay.

The `backend` module wraps the C API of NVIDIA's cuDNN library.

The `frontend` module is a Rust port of NVIDIA's [open-source C++ cuDNN Frontend library](https://github.com/NVIDIA/cudnn-frontend).
It builds operation graphs on top of the backend, infers their shapes, and picks and tunes execution plans.
Running a frontend graph still requires the cuDNN library.

## Example

Build a matrix multiplication graph with the frontend API:

```rust
use singe_cudnn::{
	backend::tensor::{Shape, TensorId, TensorSpec},
	data_type::DataType,
	frontend::{
		graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
		operation::MatmulConfig,
	},
};

fn main() -> singe_cudnn::error::Result<()> {
	let mut graph = Graph::with_config(
		GraphConfig::new().with_data_type_policy(DataTypePolicy::new().with_compute(DataType::F32)),
	);
	let a = graph.insert_tensor(
		TensorId::new(1),
		TensorSpec::new(DataType::F32, Shape::with_strides([1, 2, 3], [6, 3, 1])?),
	)?;
	let b = graph.insert_tensor(
		TensorId::new(2),
		TensorSpec::new(DataType::F32, Shape::with_strides([1, 3, 4], [12, 4, 1])?),
	)?;
	let output = graph.matmul(
		a, b, MatmulConfig::new(),
		TensorOutputConfig::new().with_data_type(DataType::F32).output_tensor(),
	)?;
	println!("output shape: {:?}", graph.shape(output)?.dimensions());
	Ok(())
}
```

This declares `[1, 2, 3] * [1, 3, 4]` with an output shape of `[1, 2, 4]`.
To run it, compile one of the supported plans and bind the graph's buffers.
The [complete matmul example](examples/frontend/matmul.rs) performs those steps and checks the result.

## Memory and execution

Raw execution is `unsafe`: the plan and every buffer must stay valid until the work finishes, or until the last replay of a captured graph.
Owned execution keeps those resources alive for you.
Capturing work into a CUDA graph does not run it.
Which plans are available depends on the GPU and the tensor layouts.

## More examples

- [Convolution](examples/frontend/convolution_fprop.rs): forward convolution planning and execution.
- [Attention](examples/frontend/fp16_sdpa.rs): FP16 scaled dot-product attention.
- [RMS normalization](examples/frontend/rms_norm.rs): build a normalization graph.
- [CUDA graphs](examples/frontend/cudagraphs.rs): capture and replay execution.
- [Serialization](examples/frontend/compiled_graph_serialize.rs): save and restore compiled plans.

Run `cargo run -p singe-cudnn --example frontend_matmul` in the repository's Nix shell.
See the [API reference](https://docs.rs/singe-cudnn).
