//! Safe Rust wrappers for the cuRAND host API, NVIDIA's random number generation library.
//! A [`Generator`](crate::generator::Generator) fills device memory on a CUDA stream, and a [`HostGenerator`](crate::generator::HostGenerator) fills host slices on the calling thread with the same algorithms.
//!
//! This crate provides:
//!
//! - **Generators:** the `XORWOW`, `MRG32k3a`, `MTGP32`, `MT19937`, and Philox pseudorandom generators, and scrambled and unscrambled Sobol quasirandom generators, with seed, offset, ordering, and dimension settings.
//! - **Distributions:** integer, uniform, normal, log-normal, and Poisson output in [`u32`], [`u64`], [`f32`], and [`f64`], through the `generate_*` methods.
//! - **Device-side support:** [`DiscreteDistribution`](crate::distribution::DiscreteDistribution), a Poisson histogram for kernels that use cuRAND's device API, and the Sobol direction vectors and scramble constants in [`quasi`].
//!
//! # Example
//!
//! Generate 1024 reproducible uniform values on the GPU and copy them to the host:
//!
//! ```rust,no_run
//! use singe_cuda::{context::Context, memory::DeviceMemory, stream::StreamOptions};
//! use singe_curand::{generator::Generator, types::RngType};
//!
//! fn main() -> anyhow::Result<()> {
//!     let context = Context::create()?;
//!     let stream = context.create_stream(StreamOptions::default())?;
//!
//!     let mut generator = Generator::create(RngType::PseudoPhilox4_32_10, &stream)?;
//!     generator.set_seed(1234)?;
//!
//!     let mut output = DeviceMemory::<f32>::create_uninit(&context, 1024)?;
//!     let values = generator.generate_uniform_f32(&mut output)?.copy_to_host_vec()?;
//!     assert!(values.iter().all(|&value| value > 0.0 && value <= 1.0));
//!     Ok(())
//! }
//! ```
//!
//! [`generate_uniform_f32`](crate::generator::Generator::generate_uniform_f32) returns once the values are ready, as a view of the now-initialized output.
//! The `unsafe` `generate_*_async` methods enqueue generation on the generator's stream without waiting.
//!
//! # Raw execution
//!
//! The `unsafe` `*_async` methods of [`generator::Generator`] check the output's context and length, order the generator's stream after the output's producers, enqueue the work, and return without waiting.
//! They are allowed during stream capture.
//! The caller must:
//!
//! - keep the generator, its stream, and the output alive, and other work away from the output, until the work completes, and through every replay of a captured graph;
//! - align the output for cuRAND's vector stores, as CUDA allocations are.
//!
//! The safe `generate_*` methods wait for the work and return a view of the initialized output.

#[cfg(not(feature = "curand_10_4"))]
compile_error!("singe-curand requires a version feature, such as `curand_10_4`");

pub mod distribution;
pub mod error;
pub mod generator;
pub mod quasi;
mod utility;

/// Generator kinds, output orderings, and Sobol direction-vector sets.
pub mod types;

pub use singe_core::LibraryVersion;
use singe_cuda::types::LibraryProperty;
pub use singe_curand_sys as sys;

use crate::error::{Error, Result, try_ffi};

/// Returns the loaded cuRAND library version.
///
/// # Errors
///
/// Returns an error if a version property query fails or a version component is negative.
pub fn version() -> Result<LibraryVersion> {
    LibraryVersion::from_properties(
        library_property(LibraryProperty::Major)?,
        library_property(LibraryProperty::Minor)?,
        library_property(LibraryProperty::Patch)?,
    )
    .ok_or_else(|| Error::out_of_range("curand version"))
}

/// Returns a cuRAND library property value.
///
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns the cuRAND status if the property query fails.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `value` is a local integer the call writes, and `property` converts to a valid native property.
    unsafe {
        try_ffi!(sys::curandGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}
