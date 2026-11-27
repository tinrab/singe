# singe-cuda

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cuda.svg)](https://crates.io/crates/singe-cuda)
[![Documentation](https://docs.rs/singe-cuda/badge.svg)](https://docs.rs/singe-cuda)
![License](https://img.shields.io/crates/l/singe-cuda.svg)

Safe Rust wrappers for the CUDA driver and runtime APIs, NVRTC, NVVM, nvJitLink, NVTX, and the CUDA profiler.

This crate provides:

- **CUDA resources:** devices, shared contexts, green contexts, streams, and events.
- **Typed memory:** device buffers and views, a pooled pinned host allocator, managed memory, stream-ordered allocations, peer copies, and virtual memory mappings.
- **Kernel execution:** runtime compilation with NVRTC and NVVM, module loading, JIT linking with nvJitLink, and typed launch methods through [`cuda_module!`](../singe-cuda-macros/README.md).
- **Asynchronous work:** composable operations with blocking or async completion, plus CUDA graph capture, construction, and replay.
- **Sharing:** interprocess memory and event handles, external memory and semaphores, and fabric multicast allocations.
- **Profiling and interoperability:** NVTX annotations, profiler control, and optional adapters for `cuda-core` and `cudarc`.

## Example

Copy values to the GPU, read them back, and print the CUDA runtime version:

```rust,no_run
use singe_cuda::{context::Context, memory::DeviceMemory};

fn main() -> anyhow::Result<()> {
    let context = Context::create()?;
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let memory = DeviceMemory::from_slice(&context, &input)?;
    assert_eq!(memory.copy_to_host_vec()?, input);
    println!("CUDA runtime {}", singe_cuda::version()?);
    Ok(())
}
```

`DeviceMemory` owns the allocation and keeps its context alive.
`from_slice` waits for the upload, and `copy_to_host_vec` waits for the readback.
The `unsafe` forms that only enqueue work are named `*_async`, such as `DeviceMemory::set_value_async` and the launch methods `cuda_module!` generates.

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

[`device_query`](./examples/device_query.rs) prints the full list of properties.

## Streams and events

Split work across streams and time it with events.

```rust,ignore
use singe_cuda::event::{EventFlags, EventRecordFlags, EventWaitFlags};

let streams = [
    context.create_stream(StreamOptions::default())?,
    context.create_stream(StreamOptions::default())?,
];
let start = context.create_event(EventFlags::DEFAULT)?;
let stop = context.create_event(EventFlags::DEFAULT)?;
start.record(&streams[0], EventRecordFlags::DEFAULT)?;
streams[1].wait_event(&start, EventWaitFlags::DEFAULT)?;

for (stream, chunk) in streams.iter().zip(&mut chunks) {
    // The demo kernel launch comes from a `cuda_module!` module.
    // SAFETY: each chunk is written by one stream and outlives the synchronization below.
    unsafe { module.scale_async(chunk, 2.0, len, &config, stream)? };
}

let done = context.create_event(EventFlags::DEFAULT)?;
done.record(&streams[1], EventRecordFlags::DEFAULT)?;
streams[0].wait_event(&done, EventWaitFlags::DEFAULT)?;
stop.record(&streams[0], EventRecordFlags::DEFAULT)?;
stop.synchronize()?;
println!("{:.3} ms", stop.elapsed_time_since(&start)?);
```

`wait_event` orders one stream after another without blocking the host.

`sync_scope` runs a closure and waits for the stream to finish before returning, even when the closure fails.
Asynchronous copies inside the closure can borrow ordinary host arrays, because the borrow checker keeps them alive and unchanged until the scope ends:

```rust,ignore
let a = [1.0_f32, 2.0, 3.0, 4.0];
let b = [10.0_f32; 4];
let mut a_device = DeviceMemory::<f32>::zeroed(&context, 4)?;
let mut b_device = DeviceMemory::<f32>::zeroed(&context, 4)?;

stream.sync_scope(|scope| {
    a_device.copy_from_host_scoped(&a, scope)?;
    b_device.copy_from_host_scoped(&b, scope)
})?;

let mut result = [0.0_f32; 4];
stream.sync_scope(|scope| b_device.copy_to_host_scoped(&mut result, scope))?;
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

```rust,ignore
use singe_cuda::{graph::GraphInstantiateFlags, stream::StreamCaptureMode};

let graph = stream.capture(StreamCaptureMode::ThreadLocal, |scope| {
    // SAFETY: the kernel checks bounds, and `values` outlives every replay.
    unsafe { module.increment_record(&mut values, len, &config, scope) }
})?;
let executable = graph.instantiate(GraphInstantiateFlags::empty())?;

for _ in 0..5 {
    // SAFETY: every replay runs on one stream, so they cannot overlap.
    unsafe { executable.launch_async(&stream)? };
}
stream.synchronize()?;
```

With `cuda_module!` macro, the generated `_record` method takes the capture scope first, then the same arguments as the `_async` launch without its stream.

Capturing records the work without running it.
See [`graph_replay`](./examples/graph_replay.rs) for the complete program.

## Memory pools

Allocate and free on a stream from a memory pool, so repeated allocations reuse the pool's memory.

Assume `pool` is a `MemoryPool` created for the device:

```rust,ignore
use singe_cuda::memory::{DeviceMemory, transfer};

let mut buffer = DeviceMemory::<u8>::zeroed_in_pool(4096, &pool, &stream)?;
buffer.set_value(7)?;
assert_eq!(transfer::copy_to_host_vec(&buffer)?, vec![7; 4096]);
buffer.release()?;
```

The [`memory_pool`](./examples/memory_pool.rs) example also creates the pool, keeps memory reserved between allocations, and trims it at the end.

## Pinned host memory

Ordinary host memory is pageable.
The OS may move it or swap it out, so the GPU cannot read it directly.
CUDA copies pageable memory through a pinned staging buffer of its own, and an asynchronous copy from pageable memory may block the calling thread until the data is staged.

Pinned (or page-locked) memory stays in physical memory at a fixed address, so the GPU reads and writes it directly.
Copies to and from it run at higher bandwidth, and an asynchronous copy returns at once, overlapping with other work.
The OS can't page pinned memory, so use it for buffers that move data to and from the GPU, not for everything.
Allocating it is also slower than ordinary allocation, which is why each context keeps a pool of it.

`Context::pinned_host` returns the allocator for that pool, a [`std::alloc::Allocator`](https://doc.rust-lang.org/std/alloc/trait.Allocator.html).
A `Vec` that uses it is a `PinnedVec`, which pinned copies accept:

```rust,ignore
use singe_cuda::memory::{DeviceMemory, PinnedVec};

let pinned = context.pinned_host();
let mut input: PinnedVec<f32> = Vec::new_in(pinned.clone());
input.extend((0..4096_u16).map(f32::from));

let mut device = DeviceMemory::<f32>::zeroed(&context, input.len())?;
stream.sync_scope(|scope| device.copy_from_pinned_scoped(&input, scope))?;

let mut output = pinned.zeroed::<f32>(input.len())?;
device.copy_to_pinned(&mut output)?;
assert_eq!(output, input);
```

A `PinnedVec` is an ordinary `Vec` otherwise.
Dropping one returns its memory to the pool without a CUDA call, so short-lived buffers are cheap, and `release_cached` gives memory nothing uses back to CUDA.
A `Vec` that fails to grow aborts the process, as with any allocator.
`with_capacity`, `from_slice`, and `zeroed` return an error instead, so use them when the size is known up front.
The pool does not know which buffers queued GPU work still uses, so keep a buffer alive until that work finishes; `sync_scope` and submissions do this for you.

The pool's memory is also mapped into the device's address space.
`PinnedVecExt::mapped_device_view` and `mapped_device_view_mut` give a kernel or copy direct access to a `PinnedVec`, with no copy to device memory first.
Every such access crosses the bus, so a view suits data the device touches once, such as a few values a kernel reports back.

See [`pinned_memory`](./examples/pinned_memory.rs) for the complete program.

## Typed kernels

With the `macros` feature, `cuda_module!` reads the kernel's parameter list and generates Rust methods with matching argument types.
Marking a pointer `read`, `write`, or `read_write` decides how the methods borrow the buffer.

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
unsafe { module.saxpy_async(2.0, &x, &mut y, 4, &config, &stream)? };
stream.synchronize()?;
assert_eq!(y.copy_to_host_vec()?, [12.0, 14.0, 16.0, 18.0]);
```

Note; writing CUDA code like this isn't necessary, obviously.
You can write it in CUDA C++ and use `cc` to build and link it with your Rust crate.

Here `x` is borrowed immutably and `y` mutably, so passing a `DeviceMemory<f64>` or the same buffer twice fails to compile.
A `write` parameter also accepts uninitialized memory, such as `DeviceMemory<MaybeUninit<f32>>`, which cannot be read back until it is marked initialized.
Each kernel gets `unsafe` launch methods with the same kernel arguments: `saxpy_async` for a chosen stream, `saxpy_record` for graph capture, `saxpy_node` for graph construction, and `saxpy_set_node_params` for updating an executable graph's node.
A pointer parameter without an element type, such as `void*`, takes a raw pointer instead of a view.

## Custom types

With the `macros` feature, derive the device representation traits for a `#[repr(C)]` struct to copy it to the GPU and pass it to kernels.
`DeviceRepr` fails to compile for a struct with implicit padding, and every derive requires each field to implement the same trait:

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

// SAFETY: the kernel checks bounds, and `stream.synchronize` waits for it.
unsafe { physics.step_async(&mut device_particles, 0.5, 3, &config, &stream)? };
stream.synchronize()?;
assert_eq!(device_particles.copy_to_host_vec()?[0].position, [0.5, 1.0]);
```

The type in `read_write(...)` is resolved inside the generated module, hence `super::`.

## Interoperability

Enable the `interop_cuda_core` or `interop_cudarc` feature to use singe-cuda alongside those crates.
The snippet uses both features:

```rust,ignore
// Hand a singe-cuda context and stream to cuda-core.
let device = context.to_cuda_core()?;
let foreign_stream = stream.to_cuda_core()?;

// Wrap a cudarc context in a singe-cuda context.
let context = Context::from_cudarc(cudarc_context)?;
```

The adapters keep the original owner alive, so neither side releases a context or stream the other is still using.

## Memory and safety

Every resource keeps its CUDA context alive.
Safe calls wait for their GPU work, `sync_scope` borrows its buffers until the stream finishes, and a submitted operation owns its buffers until its result is ready.
Nothing keeps a buffer alive for an `unsafe` `*_async` call, a generated kernel launch, or a graph launch.
The caller keeps every buffer valid, and conflicting accesses ordered, until that work finishes, including every graph replay.
Only types that implement `DeviceRepr` can be transferred, and its derive checks the layout of your own structs.

Raw kernel launches and external pointers are `unsafe`.
The caller must make sure every access stays in bounds.

## Features

- `macros`: re-exports the `DeviceRepr` derives and `cuda_module!` from `singe-cuda-macros`; `cuda_module!` also needs the default `nvrtc_13_4` feature.
- `interop_cuda_core`: adapters between this crate's contexts and streams and `cuda-core`'s.
- `interop_cudarc`: adapters between this crate's contexts and streams and `cudarc`'s.

## More examples

- [Device query](./examples/device_query.rs): print the driver and runtime versions and each GPU's properties.
- [Vector addition](./examples/vector_add.rs): compile a kernel with `cuda_module!`, add two vectors on the device, and check the result, with `--features macros`.
- [Streams and events](./examples/stream_events.rs): run work across streams and measure completion time, with `--features macros`.
- [Graph replay](./examples/graph_replay.rs): capture a kernel and replay it against the same device buffer, with `--features macros`.
- [Owned transfers](./examples/owned_transfers.rs): compose an upload and readback, then await completion.
- [Memory pool](./examples/memory_pool.rs): reuse pool-backed allocations, free them on a stream, and trim unused memory.
- [Pinned memory](./examples/pinned_memory.rs): fill a pinned vector, copy it through the device, and reuse the pool's memory.
- [Stream pool](./examples/stream_pool.rs): distribute independent jobs across streams and collect their results.
