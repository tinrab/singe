# singe-cuda-macros

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cuda-macros.svg)](https://crates.io/crates/singe-cuda-macros)
[![Documentation](https://docs.rs/singe-cuda-macros/badge.svg)](https://docs.rs/singe-cuda-macros)
![License](https://img.shields.io/crates/l/singe-cuda-macros.svg)

Derive macros for types that can be copied to the GPU, and `cuda_module!` for typed CUDA kernel launches.
The generated code names `singe_cuda` items, so use the macros through [`singe-cuda`](../singe-cuda/README.md)'s re-exports with its `macros` feature.

This crate provides:

- **`DeviceRepr`:** derives the device representation of a `#[repr(C)]` or `#[repr(transparent)]` struct, and fails to compile when the struct has implicit padding.
- **`NoUninitDeviceRepr`, `ZeroableDeviceRepr`, and `AnyBitPatternDeviceRepr`:** mark a struct as having every byte initialized, as valid when zeroed, and as valid for any bit pattern, when each of its fields is.
- **`cuda_module!`:** compiles inline CUDA C++ with NVRTC when the module is created, with optional headers and NVRTC arguments.
- **Typed launch methods:** `<kernel>_async` for a stream, `<kernel>_record` for stream capture, `<kernel>_node` for graph construction, and `<kernel>_set_node_params` for updating an executable graph.
- **Parameter access:** `read`, `write`, and `read_write` decide whether a launch method borrows a buffer immutably or mutably, and whether it accepts uninitialized memory.

## Example

Derive the device traits for a struct, then pass a buffer of it to a kernel:

```rust,ignore
use singe_cuda::{
    AnyBitPatternDeviceRepr, DeviceRepr, NoUninitDeviceRepr, ZeroableDeviceRepr,
    context::Context, cuda_module, memory::DeviceMemory, module::LaunchConfig,
    stream::StreamOptions,
};

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

fn main() -> anyhow::Result<()> {
    let context = Context::create()?;
    let stream = context.create_stream(StreamOptions::default())?;
    let particles = [Particle { position: [0.0, 0.0], velocity: [1.0, 2.0] }; 3];
    let mut device_particles = DeviceMemory::from_slice(&context, &particles)?;
    let physics = physics::Module::create(&context)?;
    let config = LaunchConfig::for_1d_grid(particles.len(), 128);

    // SAFETY: the kernel checks bounds, and `stream.synchronize` waits for it.
    unsafe { physics.step_async(&mut device_particles, 0.5, 3, &config, &stream)? };
    stream.synchronize()?;
    assert_eq!(device_particles.copy_to_host_vec()?[0].position, [0.5, 1.0]);
    Ok(())
}
```

`DeviceMemory::from_slice` needs `DeviceRepr`, and `copy_to_host_vec` needs `AnyBitPatternDeviceRepr`.
`NoUninitDeviceRepr` and `ZeroableDeviceRepr` require `DeviceRepr`, and `AnyBitPatternDeviceRepr` requires `ZeroableDeviceRepr`.
The type in `read_write(...)` is resolved inside the generated module, hence `super::`.
A pointer parameter without an element type, such as `void*`, takes a raw pointer instead of a device view.

When `singe-cuda` is renamed in `Cargo.toml`, name its path with `#[device_repr(crate = path)]` on a derive, or a `crate: path` field in `cuda_module!`.

## More examples

- [Vector addition](../singe-cuda/examples/vector_add.rs): declare and launch a typed CUDA kernel with `cuda_module!`.
- [Graph replay](../singe-cuda/examples/graph_replay.rs): record a generated launch method in a CUDA graph and replay it.
- [Streams and events](../singe-cuda/examples/stream_events.rs): launch generated methods on several streams and time them with events.

The examples belong to `singe-cuda`; run one with `cargo run -p singe-cuda --features macros --example vector_add`.
