# CUDA kernels and tuning

## Kernel families

| Category      | Module                                              | Purpose                                              |
| ------------- | --------------------------------------------------- | ---------------------------------------------------- |
| Activation    | [cuda::activation](../src/cuda/activation.rs)       | Nonlinearities and gated activations                 |
| Attention     | [cuda::attention](../src/cuda/attention.rs)         | Prefill and decode attention                         |
| Audio         | [cuda::audio](../src/cuda/audio.rs)                 | STFT, spectrograms, and audio features               |
| Cast          | [cuda::cast](../src/cuda/cast.rs)                   | Device datatype conversion                           |
| Convolution   | [cuda::conv](../src/cuda/conv.rs)                   | Convolution and layout helpers                       |
| Creation      | [cuda::creation](../src/cuda/creation.rs)           | Fill, zeros, ones, and ranges                        |
| Elementwise   | [cuda::elementwise](../src/cuda/elementwise.rs)     | Binary and ternary operations, clamps, and selects   |
| Embedding     | [cuda::embedding](../src/cuda/embedding.rs)         | Token embedding lookups                              |
| FFT           | [cuda::fft](../src/cuda/fft.rs)                     | Complex and real transforms, normalization           |
| Fused         | [cuda::fused](../src/cuda/fused.rs)                 | Combined inference operations                        |
| Gather        | [cuda::gather](../src/cuda/gather.rs)               | Indexed gathers and row selection                    |
| Mask          | [cuda::mask](../src/cuda/mask.rs)                   | Boolean and causal masks                             |
| Matmul        | [cuda::matmul](../src/cuda/matmul.rs)               | GEMM, batched products, and matrix-vector products   |
| MoE           | [cuda::moe](../src/cuda/moe.rs)                     | Expert routing and alignment                         |
| Normalization | [cuda::normalization](../src/cuda/normalization.rs) | RMSNorm, LayerNorm, GroupNorm, and sparsemax         |
| Pooling       | [cuda::pooling](../src/cuda/pooling.rs)             | Pooling over device tensors                          |
| Positional    | [cuda::positional](../src/cuda/positional.rs)       | RoPE and positional embeddings                       |
| Quantization  | [cuda::quantization](../src/cuda/quantization.rs)   | Quantize/dequantize, INT4, FP8, and quantized matmul |
| Recurrent     | [cuda::recurrent](../src/cuda/recurrent.rs)         | Recurrent and state-space preprocessing              |
| Reduction     | [cuda::reduction](../src/cuda/reduction.rs)         | Sum, extrema, norms, and axis reductions             |
| Scalar        | [cuda::scalar](../src/cuda/scalar.rs)               | Scalar arithmetic and comparisons                    |
| Shape         | [cuda::shape](../src/cuda/shape.rs)                 | Copy, transpose, pad, slice, and tile                |
| Softmax       | [cuda::softmax](../src/cuda/softmax.rs)             | Softmax and log-sum-exp                              |
| Unary         | [cuda::unary](../src/cuda/unary.rs)                 | Single-input math and predicates                     |

## Select features

Enable a `kernel_*` feature for each CUDA operation family and a `dtype_*` feature for each input datatype.
For example, `kernel_matmul` enables `cuda::matmul` and its CUDA/cuTile dependencies.
The default datatype is `dtype_f32`; add `dtype_f16` or `dtype_bf16` for half precision.
Individual operations support different subsets of these types.

| Feature                                                                    | Enables                                                             |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `kernel_*`                                                                 | A CUDA operation family, such as `kernel_attention` or `kernel_fft` |
| `dtype_f32`, `dtype_f16`, `dtype_bf16`, `dtype_f64`, `dtype_f8`            | Floating-point APIs                                                 |
| `dtype_i8`, `dtype_i32`, `dtype_i64`, `dtype_u8`, `dtype_u32`, `dtype_u64` | Integer APIs                                                        |
| `kernels_all`, `dtypes_all`                                                | Every operation family or datatype, respectively                    |
| `full`                                                                     | All CUDA kernels and datatypes                                      |
| `serde`                                                                    | Serialization of tuning keys, reports, and tuning values            |

We have CPU kernels that don't need CUDA.

## Launch a CUDA kernel

With `kernel_matmul` enabled, multiply two row-major matrices:

```rust,no_run
use singe_cuda::{context::Context, device::Device, memory::DeviceMemory};
use singe_kernel::cuda::matmul::{self, MatmulConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let context = Context::retain_primary_for_device(Device::new(0))?;
	let stream = context.create_stream()?;
	let a = DeviceMemory::from_slice(&context, &[1.0_f32, 2.0, 3.0, 4.0])?;
	let b = DeviceMemory::from_slice(&context, &[5.0_f32, 6.0, 7.0, 8.0])?;
	let mut output = DeviceMemory::<f32>::create_uninit(&context, 4)?;
	let config = MatmulConfig::row_major(2, 2, 2);

	stream.sync_scope(|_| {
		// SAFETY: the initialized inputs and disjoint output match the layout
		// and remain alive through completion, including on errors.
		Ok(unsafe { matmul::matmul_f32(&stream, &mut output, &a, &b, config) })
	})??; // The first `?` handles the scope's CUDA error; the second handles the kernel's result.

	// SAFETY: the completed product wrote all four output elements.
	let output = unsafe { output.assume_init() };
	assert_eq!(output.copy_to_host_vec()?, [19.0, 22.0, 43.0, 50.0]);
	Ok(())
}
```

`MatmulConfig` describes dimensions, strides, transposes, and tuning.
`sync_scope` waits before returning or unwinding, so allocations declared outside its callback survive pending work.

## Tuning

For kernels with tunable configurations:

| Configuration             | Behavior                                                             |
| ------------------------- | -------------------------------------------------------------------- |
| `.tune(options)`          | Tune on a cache miss and store the winner in the process-local cache |
| `.cached_tuning()`        | Require an existing cache entry; suitable for capture after warmup   |
| `.explicit_tuning(value)` | Launch with a supplied tuning value without consulting the cache     |
| `.retune(options)`        | Tune again and replace the cached entry                              |

Tuning runs multiple candidates against the supplied buffers.
Warm up and populate the cache before graph capture.

### Save a tuning result

The `tune_*` functions return a `TuneReport` containing the selected value and its key.

Using the inputs, stream, context, and configuration from the launch example:

```rust,ignore
use singe_kernel::cuda::autotune::TuneOptions;

let mut output = DeviceMemory::<f32>::create_uninit(&context, 4)?;
let report = stream.sync_scope(|_| {
	// SAFETY: every candidate uses the same valid inputs and disjoint output,
	// with all allocations retained until the scope drains.
	Ok(unsafe {
		matmul::tune_matmul_f32(
			&stream, &mut output, &a, &b, config, TuneOptions::default(),
		)
	})
})??;
std::fs::write("matmul-tuning.json", serde_json::to_string_pretty(&report)?)?;
```

Enable `serde` on `singe-kernel` and add `serde_json` to serialize reports.

### Reuse a saved result

Rebuild the expected `TuneKey` for the current operation, backend, device, and problem, then compare it with the saved key.

For matmul, the problem key includes the datatype, dimensions, row strides, and transpose flags.
Use `TuneKey::create`, `DeviceFingerprint::create`, and `TuneProblemKey::Matmul` from the [autotuning API](../src/cuda/autotune.rs).

With that `expected_key` and the current `config`:

```rust,ignore
use singe_kernel::cuda::{autotune::TuneReport, matmul::MatmulTuning};

let report: TuneReport<MatmulTuning> =
	serde_json::from_str(&std::fs::read_to_string("matmul-tuning.json")?)?;
assert_eq!(report.key, expected_key, "saved tuning does not match this launch");
let config = config.explicit_tuning(report.best);
```

Pass this configuration to the kernel launch.

Loading a report does not populate the process-local cache.
Retune when the key or tuning schema changes.

## Raw cuTile APIs

`cuda::cutile::kernel` exposes the cuTile kernels directly, without Singe's buffer and launch wrappers.

For example, fill a device buffer with an explicit grid:

```rust,no_run
use cutile::cuda_async::device_operation::DeviceOp;
use singe_cuda::{context::Context, device::Device, memory::DeviceMemory};
use singe_kernel::cuda::cutile::{DevicePointer, TileKernel, kernel::common};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let context = Context::retain_primary_for_device(Device::new(0))?;
	let stream = context.create_stream()?;
	let cuda_stream = stream.to_cuda_core()?;
	let output = DeviceMemory::<f32>::create_uninit(&context, 4)?;
	// SAFETY: the allocation has room for four f32 values.
	let pointer = unsafe {
		DevicePointer::<f32>::from_cu_deviceptr(output.as_mut_ptr() as _)
	};

	stream.sync_scope(|_| {
		// SAFETY: the kernel checks bounds, and output stays alive and exclusive
		// until this scope completes the launch.
		Ok(unsafe {
			common::fill_f32(pointer, 2.0, 4)
				.grid((1, 1, 1))
				.async_on(&cuda_stream)
		})
	})??;
	// SAFETY: the completed fill initialized every element.
	let output = unsafe { output.assume_init() };
	assert_eq!(output.copy_to_host_vec()?, [2.0; 4]);
	Ok(())
}
```

Enable `kernel_creation` and `dtype_f32`, and add `cutile = "0.4"` for this example.
