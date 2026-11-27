//! Safe Rust wrappers for NVIDIA cuFFT, cuFFTXt, and the cuFFT device API, for fast Fourier transforms on one or several GPUs.
//!
//! This crate provides:
//!
//! - **Plans:** one-, two-, and three-dimensional transforms and batched strided layouts, through [`plan_1d`](crate::plan::Plan::plan_1d), [`plan_2d`](crate::plan::Plan::plan_2d), [`plan_3d`](crate::plan::Plan::plan_3d), and [`plan_many`](crate::plan::Plan::plan_many).
//! - **Transforms:** complex-to-complex, real-to-complex, and complex-to-real, in single and double precision.
//! - **Workspace control:** cuFFT allocates the workspace during planning by default; [`set_auto_allocation(false)`](crate::plan::Plan::set_auto_allocation) with [`set_work_area`](crate::plan::Plan::set_work_area) installs one you own.
//! - **cuFFTXt:** element types given as CUDA data types, link-time callbacks, and transforms split across several GPUs, in [`xt`].
//! - **Device functions:** the experimental [`device`] module describes FFTs and extracts their LTO-IR or fatbinary code for linking into your own kernels with nvJitLink.
//!
//! # Example
//!
//! Transform eight complex values in place, apply the inverse, and check the result:
//!
//! ```rust,no_run
//! use singe_cuda::{context::Context, memory::DeviceMemory, stream::StreamOptions, types::Complex32};
//! use singe_cufft::{
//!     plan::Plan,
//!     types::{Direction, TransformType},
//! };
//!
//! fn main() -> anyhow::Result<()> {
//!     let context = Context::create()?;
//!     let stream = context.create_stream(StreamOptions::default())?;
//!
//!     let input = [Complex32::new(1.0, 0.0); 8];
//!     let mut values = DeviceMemory::from_slice(&context, &input)?;
//!
//!     let mut plan = Plan::create(&context)?;
//!     plan.plan_1d(input.len(), TransformType::ComplexToComplex, 1)?;
//!     plan.execute_in_place(&mut values, Direction::Forward, &stream)?;
//!     plan.execute_in_place(&mut values, Direction::Inverse, &stream)?;
//!
//!     // cuFFT does not normalize the inverse transform, so the round trip scales each value by eight.
//!     let output = values.copy_to_host_vec()?;
//!     assert!(output.iter().all(|value| (value - Complex32::new(8.0, 0.0)).norm() < 1.0e-3));
//!     Ok(())
//! }
//! ```
//!
//! A [`Plan`](crate::plan::Plan) describes a transform, manages its workspace, and runs it on a stream.
//!
//! [`execute_in_place`](crate::plan::Plan::execute_in_place) returns once the transform has finished on `stream`.
//! When a plan runs on a different stream than last time, the new stream first waits for the work queued on the previous one.
//! [`Plan::execute_async`](crate::plan::Plan::execute_async) and [`execute_in_place_async`](crate::plan::Plan::execute_in_place_async) are `unsafe` and only enqueue the transform, and the `_scoped` forms enqueue it into a [`StreamScope`](singe_cuda::stream::StreamScope).
//!
//! # Raw execution
//!
//! The `unsafe` `*_async` functions check contexts, element types, operand lengths against the configured transform, and alignment, order the transform on the given stream after its operands' producers, enqueue it, and return without waiting.
//! They are allowed during stream capture.
//! The caller must:
//!
//! - keep the plan, its work area, callback state, and every operand alive and unchanged, without reconfiguring the plan, until the work completes, including after a failed submission, and through every replay of a captured graph;
//! - initialize every sample the transform reads;
//! - hold writable operands, and the input of a complex-to-real transform, exclusively until the work completes.
//!
//! A plan's work area serves one transform at a time, so when an execution moves the plan to another stream, the new stream first waits for the work queued on the previous one.
//! The safe forms wait for completion, and the `_scoped` forms record their borrows in a [`StreamScope`](singe_cuda::stream::StreamScope).

#[cfg(not(feature = "cufft_12_4"))]
compile_error!("singe-cufft requires a version feature, such as `cufft_12_4`");

pub mod device;

/// cuFFT status codes and errors from planning, execution, and CUDA operations.
pub mod error;

/// Single-GPU FFT planning, workspace configuration, and transform submission.
pub mod plan;

/// Serialized test setup for cuFFT on CUDA device 0.
#[cfg(feature = "testing")]
pub mod testing;

/// Transform directions, element types, and native plan properties.
pub mod types;

pub(crate) mod utility;

pub mod xt;

pub use singe_core::LibraryVersion;
use singe_cuda::types::LibraryProperty;
pub use singe_cufft_sys as sys;

use crate::error::{Error, Result, try_ffi};

/// Returns the loaded cuFFT library version from its major, minor, and patch properties.
///
/// # Errors
///
/// Returns an error if cuFFT cannot report one of the version properties or a version component is negative.
pub fn version() -> Result<LibraryVersion> {
    LibraryVersion::from_properties(
        library_property(LibraryProperty::Major)?,
        library_property(LibraryProperty::Minor)?,
        library_property(LibraryProperty::Patch)?,
    )
    .ok_or_else(|| Error::out_of_range("cufft version"))
}

/// Returns the requested cuFFT library property.
/// See [`LibraryProperty`] for supported properties.
///
/// # Errors
///
/// Returns an error if cuFFT cannot report the requested property.
pub fn library_property(property: LibraryProperty) -> Result<i32> {
    let mut value = 0;
    // SAFETY: `value` is a local integer the call writes, and `property` converts to a valid native property.
    unsafe {
        try_ffi!(sys::cufftGetProperty(property.into(), &raw mut value))?;
    }
    Ok(value)
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_it_works() -> Result<()> {
        let _test = setup_context()?;
        let version = version()?;
        assert!(
            version.major() >= 11,
            "cuFFT major version {version} is implausible"
        );
        let mut packed = 0;
        // SAFETY: `packed` is a valid output location.
        unsafe {
            try_ffi!(sys::cufftGetVersion(&raw mut packed))?;
        }
        let expected = version.major() * 1000 + version.minor() * 100 + version.patch();
        assert_eq!(u32::try_from(packed).ok(), Some(expected));
        Ok(())
    }
}
