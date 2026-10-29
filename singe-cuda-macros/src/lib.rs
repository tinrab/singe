//! Derives for CUDA device representations and macros for typed kernel launch modules.

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
/// `read`, `write` and `read_write` select initialized input, write-only output and initialized mutable views.
/// Use `value(RustType)` for by-value struct mappings and `read(RustType)` or writable equivalents for custom pointer elements.
/// Rust by-value types must implement `NoUninitDeviceRepr` and match the CUDA field layout.
/// Each kernel gets a public signature and `prepare_<name>` method.
/// Typed view methods use the kernel name; raw pointer methods append `_raw`.
/// Executable-node parameter updates take `LaunchGeometry` and preserve existing launch attributes.
/// Export aliases retain independent annotations; generated name collisions require distinct aliases.
/// Submission remains unsafe because ABI checks do not establish semantic bounds, access ordering, storage lifetimes or output initialization coverage.
///
/// Expansion parses the CUDA source with tree-sitter and emits a Rust module
/// with a `create` constructor plus typed wrappers for exported kernels.
/// CUDA compilation, PTX/cubin generation, and lowered kernel-name resolution happen through NVRTC when the generated module is created.
#[proc_macro]
pub fn cuda_module(input: TokenStream) -> TokenStream {
    match cuda_module::expand_module(input.into()) {
        Ok(output) => output.into(),
        Err(error) => error.to_compile_error().into(),
    }
}
