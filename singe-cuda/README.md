# singe-cuda

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cuda.svg)](https://crates.io/crates/singe-cuda)
[![Documentation](https://docs.rs/singe-cuda/badge.svg)](https://docs.rs/singe-cuda)
![License](https://img.shields.io/crates/l/singe-cuda.svg)

Safe Rust wrappers for the CUDA driver and runtime APIs, including runtime compilation of CUDA C++ and NVTX annotations.

This crate provides:

- **CUDA resources:** devices, shared contexts, streams, and events.
- **Typed memory:** device buffers and views, pinned and managed memory, stream-ordered allocations, and peer copies.
- **Kernel execution:** runtime compilation with NVRTC, module loading, JIT linking, and typed launch methods through [`cuda_module!`](../singe-cuda-macros/README.md).
- **Asynchronous work:** composable operations with blocking or async completion, plus CUDA graph capture and replay.
- **Profiling and interoperability:** NVTX annotations and optional adapters for `cuda-core` and `cudarc`.

## Example

Allocate device memory, launch a kernel that fills it, and read back the result:

```rust
use singe_cuda::{context::Context, cuda_module, memory::DeviceMemory, module::LaunchConfig};

cuda_module! {
	pub mod kernels {
		source: r#"
			extern "C" __global__ void fill(unsigned int* values, int len) {
				int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
				if (i < len) values[i] = i * 2;
			}
		"#,
		exports: {
			fill { parameters: { values: write } },
		},
	}
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let context = Context::create()?;
	let mut values = DeviceMemory::<u32>::zeroed(&context, 4)?;
	let module = kernels::Module::create(&context)?;
	let config = LaunchConfig::for_1d_grid(values.len(), 128);

	// SAFETY: the kernel checks bounds, and the buffer and module stay alive
	// until the blocking readback completes work on the default stream.
	unsafe { module.fill(&config, &mut values, 4)?; }

	assert_eq!(values.copy_to_host_vec()?, [0, 2, 4, 6]);
	Ok(())
}
```

`DeviceMemory` owns the GPU allocation, and `cuda_module!` generates typed kernel launch methods.
`LaunchConfig` sets the grid and block dimensions.
The readback waits for the kernel to finish before returning host values.

For a complete vector addition example, see [`vector_add`](examples/vector_add.rs).

## Device information

Query the installed CUDA versions and the properties of each GPU:

```rust,ignore
use singe_cuda::{device::Device, driver, runtime};

println!("driver {}, runtime {}", driver::version()?, runtime::version()?);
for ordinal in 0..Device::count()? {
	let properties = Device::new(ordinal).properties()?;
	println!(
		"{ordinal}: {} (compute {}.{}, {} MiB, {} SMs)",
		properties.name,
		properties.major,
		properties.minor,
		properties.total_global_mem / (1024 * 1024),
		properties.multi_processor_count,
	);
}
```

[`device_query`](examples/device_query.rs) prints the full list of properties.

## Streams and events

Split work across streams and time it with events.

```rust,ignore
use singe_cuda::event::EventRecordFlags;

let streams = [context.create_stream()?, context.create_stream()?];
let start = context.create_event()?;
let stop = context.create_event()?;
start.record(&streams[0], EventRecordFlags::DEFAULT)?;
streams[1].wait_event(&start)?;

for (stream, chunk) in streams.iter().zip(&mut chunks) {
	// SAFETY: each chunk is written by one stream and outlives the synchronization below.
	unsafe { module.scale_on(&config, stream, chunk, 2.0, len)? };
}

let done = context.create_event()?;
done.record(&streams[1], EventRecordFlags::DEFAULT)?;
streams[0].wait_event(&done)?;
stop.record(&streams[0], EventRecordFlags::DEFAULT)?;
stop.synchronize()?;
println!("{:.3} ms", stop.elapsed_time_since(&start)?);
```

`wait_event` orders one stream after another without blocking the host.

`sync_scope` runs a closure and waits for the stream to finish before returning.
Asynchronous copies inside the closure can borrow ordinary host arrays, because the borrow checker keeps them alive and unchanged until the scope ends:

```rust,ignore
let a = [1.0_f32, 2.0, 3.0, 4.0];
let b = [10.0_f32; 4];
let mut a_device = DeviceMemory::<f32>::zeroed(&context, 4)?;
let mut b_device = DeviceMemory::<f32>::zeroed(&context, 4)?;

stream.sync_scope(|scope| {
	a_device.copy_from_host_async(&a, scope)?;
	b_device.copy_from_host_async(&b, scope)
})?;

let mut result = [0.0_f32; 4];
stream.sync_scope(|scope| b_device.copy_to_host_async(&mut result, scope))?;
assert_eq!(result, b);
```

To run host code once a stream reaches a point, enqueue a closure with `launch_host_func`:

```rust,ignore
use std::sync::mpsc;

let (sender, receiver) = mpsc::channel();
stream.launch_host_func(move || sender.send("done").unwrap())?;
assert_eq!(receiver.recv()?, "done");
```

The closure runs on a CUDA thread and must not call CUDA itself.

## Asynchronous work

Compose an upload and a readback into one operation, then await it:

```rust,ignore
use singe_cuda::submission::{Operation, transfer::{DeviceValue, Upload}};

let upload = Upload::create(&context, &input)?;
drop(input); // the upload already copied it into pinned staging memory
let output = upload.then(DeviceValue::readback).submit_on(&stream)?.await?;
```

The operation owns its staging and device buffers until the result is ready.
To spread independent jobs over several streams, submit them to a `StreamPool`:

```rust,ignore
use singe_cuda::submission::StreamPool;

let pool = StreamPool::create(&context, 2)?;
let mut submissions = Vec::new();
for job in 0..8_u32 {
	let operation = Upload::create(&context, &vec![job; 4096])?.then(DeviceValue::readback);
	submissions.push((job, pool.submit(operation)?));
}
// Every job is submitted before the first wait, so the streams run them concurrently.
for (job, submission) in submissions {
	assert_eq!(submission.wait()?, vec![job; 4096]);
}
```

## CUDA graphs

Record kernel launches once and replay them with little CPU overhead.
The generated `_record` method takes the same arguments as the regular launch:

```rust,ignore
use singe_cuda::stream::StreamCaptureMode;

let graph = stream.capture(StreamCaptureMode::ThreadLocal, |scope| {
	// SAFETY: the kernel checks bounds, and `values` outlives every replay.
	unsafe { module.increment_record(scope, &config, &mut values, len) }
})?;
let executable = graph.instantiate()?;

for _ in 0..5 {
	// SAFETY: every replay runs on one stream, so they cannot overlap.
	unsafe { executable.launch(&stream)? };
}
stream.synchronize()?;
```

Capturing records the work without running it.
See [`graph_replay`](examples/graph_replay.rs) for the complete example.

## Memory pools

Allocate and free on a stream from a memory pool, so repeated allocations reuse the pool's memory.
Assume `pool` is a `MemoryPool` created for the device:

```rust,ignore
use singe_cuda::memory::{AsyncDeviceMemory, transfer};

let mut buffer = AsyncDeviceMemory::<u8>::zeroed_in_pool(4096, &pool, &stream)?;
buffer.set_value_async(7)?;
assert_eq!(transfer::copy_to_host_vec(&buffer)?, vec![7; 4096]);
buffer.free_async()?;
```

The [`memory_pool`](examples/memory_pool.rs) example also creates the pool, keeps memory reserved between allocations, and trims it at the end.

## Typed kernels

`cuda_module!` reads the kernel's parameter list and generates a Rust method with matching argument types.
Marking a pointer `read`, `write`, or `read_write` decides how the method borrows the buffer:

```rust,ignore
cuda_module! {
	pub mod kernels {
		source: r#"
			extern "C" __global__ void saxpy(float a, const float* x, float* y, int len) {
				int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
				if (i < len) y[i] = a * x[i] + y[i];
			}
		"#,
		exports: {
			saxpy { parameters: { x: read, y: read_write } },
		},
	}
}

let x = DeviceMemory::from_slice(&context, &[1.0_f32, 2.0, 3.0, 4.0])?;
let mut y = DeviceMemory::from_slice(&context, &[10.0_f32; 4])?;
let module = kernels::Module::create(&context)?;
let config = LaunchConfig::for_1d_grid(x.len(), 128);

// SAFETY: the kernel checks bounds, and `stream.synchronize` waits for it.
unsafe { module.saxpy_on(&config, &stream, 2.0, &x, &mut y, 4)? };
stream.synchronize()?;
assert_eq!(y.copy_to_host_vec()?, [12.0, 14.0, 16.0, 18.0]);
```

Here `x` is borrowed immutably and `y` mutably, so passing a `DeviceMemory<f64>` or the same buffer twice fails to compile.
A `write` parameter also accepts uninitialized memory, such as `DeviceMemory<MaybeUninit<f32>>`, which cannot be read back until it is marked initialized.
Each kernel gets three launch methods with the same arguments: `saxpy` for the default stream, `saxpy_on` for a chosen stream, and `saxpy_record` for graph capture.

## Custom types

Derive the device representation traits for a `#[repr(C)]` struct to copy it to the GPU and pass it to kernels.
The derives check that the struct has no padding and that every field is safe to copy:

```rust,ignore
use singe_cuda::{AnyBitPatternDeviceRepr, DeviceRepr, NoUninitDeviceRepr, ZeroableDeviceRepr};

#[repr(C)]
#[derive(Clone, Copy, DeviceRepr, NoUninitDeviceRepr, ZeroableDeviceRepr, AnyBitPatternDeviceRepr)]
pub struct Particle {
	pub position: [f32; 2],
	pub velocity: [f32; 2],
}

cuda_module! {
	pub mod physics {
		source: r#"
			struct Particle { float position[2]; float velocity[2]; };

			extern "C" __global__ void step(Particle* particles, float dt, int len) {
				int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
				if (i < len) {
					particles[i].position[0] += particles[i].velocity[0] * dt;
					particles[i].position[1] += particles[i].velocity[1] * dt;
				}
			}
		"#,
		exports: {
			step { parameters: { particles: read_write(super::Particle) } },
		},
	}
}

let particles = [Particle { position: [0.0, 0.0], velocity: [1.0, 2.0] }; 3];
let mut device_particles = DeviceMemory::from_slice(&context, &particles)?;
let physics = physics::Module::create(&context)?;
let config = LaunchConfig::for_1d_grid(particles.len(), 128);

// SAFETY: the kernel checks bounds, and the readback waits for it.
unsafe { physics.step(&config, &mut device_particles, 0.5, 3)? };
assert_eq!(device_particles.copy_to_host_vec()?[0].position, [0.5, 1.0]);
```

The type in `read_write(...)` is resolved inside the generated module, hence `super::`.

## Interoperability

Enable the `interop_cuda_core` or `interop_cudarc` feature to use singe-cuda alongside those crates:

```rust,ignore
// Hand a singe-cuda context and stream to cuda-core.
let device = context.to_cuda_core()?;
let foreign_stream = stream.to_cuda_core()?;

// Wrap a cudarc context in a singe-cuda context.
let context = Context::from_cudarc(cudarc_context)?;
```

The adapters keep the original owner alive, so neither side releases a context or stream the other is still using.

## Memory and safety

Every resource keeps its CUDA context alive, and pending work keeps the buffers it uses alive.
Only types that are valid to copy to and from the GPU can be transferred, and a derive macro checks the layout of your own structs.

Raw kernel launches and external pointers are `unsafe`.
The caller must make sure every access stays in bounds and every buffer stays valid until the work finishes.

## More examples

- [Device query](examples/device_query.rs): inspect GPU properties.
- [Streams and events](examples/stream_events.rs): run work across streams and measure completion time.
- [Graph replay](examples/graph_replay.rs): capture a kernel and replay it against the same device buffer.
- [Owned transfers](examples/owned_transfers.rs): compose an upload and readback, then await completion.
- [Memory pool](examples/memory_pool.rs): reuse pool-backed allocations, free them on a stream, and trim unused memory.
- [Stream pool](examples/stream_pool.rs): distribute independent jobs across streams and collect their results.

See the [API reference](https://docs.rs/singe-cuda).
