//! Derive macros for types that can be copied to the GPU, and [`cuda_module!`] for typed CUDA kernel launches.
//! The generated code names `singe_cuda` items, so use the macros through `singe-cuda`'s re-exports with its `macros` feature.
//!
//! This crate provides:
//!
//! - **[`DeviceRepr`]:** derives the device representation of a `#[repr(C)]` or `#[repr(transparent)]` struct, and fails to compile when the struct has implicit padding.
//! - **[`NoUninitDeviceRepr`], [`ZeroableDeviceRepr`], and [`AnyBitPatternDeviceRepr`]:** mark a struct as having every byte initialized, as valid when zeroed, and as valid for any bit pattern, when each of its fields is.
//! - **[`cuda_module!`]:** compiles inline CUDA C++ with NVRTC when the module is created, with optional headers and NVRTC arguments.
//! - **Typed launch methods:** `<kernel>_async` for a stream, `<kernel>_record` for stream capture, `<kernel>_node` for graph construction, and `<kernel>_set_node_params` for updating an executable graph.
//! - **Parameter access:** `read`, `write`, and `read_write` decide whether a launch method borrows a buffer immutably or mutably, and whether it accepts uninitialized memory.
//!
//! # Example
//!
//! Derive the device traits for a struct, then pass a buffer of it to a kernel:
//!
//! ```rust,ignore
//! use singe_cuda::{
//!     AnyBitPatternDeviceRepr, DeviceRepr, NoUninitDeviceRepr, ZeroableDeviceRepr,
//!     context::Context, cuda_module, memory::DeviceMemory, module::LaunchConfig,
//!     stream::StreamOptions,
//! };
//!
//! #[repr(C)]
//! #[derive(Clone, Copy, DeviceRepr, NoUninitDeviceRepr, ZeroableDeviceRepr, AnyBitPatternDeviceRepr)]
//! pub struct Particle {
//!     pub position: [f32; 2],
//!     pub velocity: [f32; 2],
//! }
//!
//! cuda_module! {
//!     pub mod physics {
//!         source: r#"
//!             struct Particle { float position[2]; float velocity[2]; };
//!
//!             extern "C" __global__ void step(Particle* particles, float dt, int len) {
//!                 int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
//!                 if (i < len) {
//!                     particles[i].position[0] += particles[i].velocity[0] * dt;
//!                     particles[i].position[1] += particles[i].velocity[1] * dt;
//!                 }
//!             }
//!         "#,
//!         exports: {
//!             step { parameters: { particles: read_write(super::Particle) } },
//!         },
//!     }
//! }
//!
//! fn main() -> anyhow::Result<()> {
//!     let context = Context::create()?;
//!     let stream = context.create_stream(StreamOptions::default())?;
//!     let particles = [Particle { position: [0.0, 0.0], velocity: [1.0, 2.0] }; 3];
//!     let mut device_particles = DeviceMemory::from_slice(&context, &particles)?;
//!     let physics = physics::Module::create(&context)?;
//!     let config = LaunchConfig::for_1d_grid(particles.len(), 128);
//!
//!     // SAFETY: the kernel checks bounds, and `stream.synchronize` waits for it.
//!     unsafe { physics.step_async(&mut device_particles, 0.5, 3, &config, &stream)? };
//!     stream.synchronize()?;
//!     assert_eq!(device_particles.copy_to_host_vec()?[0].position, [0.5, 1.0]);
//!     Ok(())
//! }
//! ```
//!
//! `DeviceMemory::from_slice` needs `DeviceRepr`, and `copy_to_host_vec` needs `AnyBitPatternDeviceRepr`.
//! `NoUninitDeviceRepr` and `ZeroableDeviceRepr` require `DeviceRepr`, and `AnyBitPatternDeviceRepr` requires `ZeroableDeviceRepr`.
//! The type in `read_write(...)` is resolved inside the generated module, hence `super::`.
//! A pointer parameter without an element type, such as `void*`, takes a raw pointer instead of a device view.
//!
//! When `singe-cuda` is renamed in `Cargo.toml`, name its path with `#[device_repr(crate = path)]` on a derive, or a `crate: path` field in [`cuda_module!`].

mod cuda_module;
mod device_repr;

use proc_macro::TokenStream;

/// Derives a transferable device representation for a struct without implicit padding.
///
/// Requires `repr(C)` or `repr(transparent)` and transferable field types.
/// Generic layouts are checked when a memory API evaluates `DeviceRepr::LAYOUT`.
/// Use `#[device_repr(crate = path)]` when `singe-cuda` has been renamed.
#[proc_macro_derive(DeviceRepr, attributes(device_repr))]
pub fn derive_device_repr(input: TokenStream) -> TokenStream {
    derive_representation(input, device_repr::Representation::Transfer)
}

/// Derives that every byte of a valid value is initialized.
///
/// Requires `DeviceRepr` and fields implementing `NoUninitDeviceRepr`.
/// `MaybeUninit` fields cannot satisfy this contract.
#[proc_macro_derive(NoUninitDeviceRepr, attributes(device_repr))]
pub fn derive_no_uninit_device_repr(input: TokenStream) -> TokenStream {
    derive_representation(input, device_repr::Representation::NoUninit)
}

/// Derives valid zero initialization for a struct whose fields are zeroable.
///
/// The struct must also implement `DeviceRepr`.
#[proc_macro_derive(ZeroableDeviceRepr, attributes(device_repr))]
pub fn derive_zeroable_device_repr(input: TokenStream) -> TokenStream {
    derive_representation(input, device_repr::Representation::Zeroable)
}

/// Derives unrestricted bit validity for a struct whose fields accept every bit pattern.
///
/// The struct must also implement `ZeroableDeviceRepr`.
#[proc_macro_derive(AnyBitPatternDeviceRepr, attributes(device_repr))]
pub fn derive_any_bit_pattern_device_repr(input: TokenStream) -> TokenStream {
    derive_representation(input, device_repr::Representation::AnyBitPattern)
}

fn derive_representation(
    input: TokenStream,
    representation: device_repr::Representation,
) -> TokenStream {
    match device_repr::expand(input.into(), representation) {
        Ok(output) => output.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

/// Generates a typed kernel-launch module that compiles CUDA source with NVRTC at runtime.
///
/// The macro accepts inline CUDA source, selected exports, inline headers and compiler arguments:
///
/// ```ignore
/// cuda_module! {
///     pub mod kernels {
///         source: r#"extern "C" __global__ void scale(float* values, int len) { /* ... */ }"#,
///         exports: {
///             scale {
///                 parameters: { values: read_write },
///                 requirements: singe_cuda::kernel::launch::LaunchRequirements {
///                     rank: singe_cuda::kernel::launch::LaunchRank::One,
///                     ..singe_cuda::kernel::launch::LaunchRequirements::DEFAULT
///                 },
///             },
///         },
///         compile: { nvrtc_args: ["--std=c++20"] },
///     }
/// }
/// ```
///
/// An optional `crate: path` field names `singe_cuda` for the generated code, such as `crate: ::my_crate::cuda` when it is re-exported; the path resolves inside the generated module, so start it with `::` or `crate`.
///
/// `read`, `write` and `read_write` select initialized input, write-only output and initialized mutable views.
/// Use `value(RustType)` for by-value struct mappings and `read(RustType)` or writable equivalents for custom pointer elements.
/// Rust by-value types must implement `NoUninitDeviceRepr` and match the CUDA field layout.
/// Each kernel gets a public signature, a `prepare_<name>` method, and one method per entry point: `<name>_async(.., config, stream)`, `<name>_record(.., config, scope)`, `<name>_node`, and `<name>_set_node_params`.
/// Typed pointer parameters take device views; pointers without an element type or with several levels take raw pointers.
/// Executable-node parameter updates take `LaunchGeometry` and preserve existing launch attributes.
/// Export aliases retain independent annotations; generated name collisions require distinct aliases.
/// Submission remains unsafe because ABI checks do not establish semantic bounds, access ordering, storage lifetimes or output initialization coverage.
///
/// Expansion parses the CUDA source with tree-sitter and emits a Rust module with the source description, a `create` constructor, and typed wrappers for exported kernels.
/// `create` calls `singe_cuda::nvrtc::compile_cached`, which compiles the source with NVRTC once per context and resolves the kernels' lowered names.
#[proc_macro]
pub fn cuda_module(input: TokenStream) -> TokenStream {
    match cuda_module::expand_module(input.into()) {
        Ok(output) => output.into(),
        Err(error) => error.to_compile_error().into(),
    }
}
