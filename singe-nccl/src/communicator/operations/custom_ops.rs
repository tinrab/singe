use std::ptr;

use singe_cuda::view::DeviceView;

use crate::{
    communicator::{Communicator, CustomReductionOperator},
    error::{Error, Result},
    sys, try_ffi,
    types::{DataTypeLike, ScalarResidence},
};

impl Communicator {
    /// Creates a reduction operator that pre-multiplies input values by `scalar` before reducing them with peer values via summation.
    /// Both the input values and `scalar` use the data type `T`.
    /// Use the returned operator only with collectives launched on this communicator and with the same `T`.
    /// `residence` indicates whether `scalar` is dereferenced immediately by the
    /// host before this method returns ([`ScalarResidence::HostImmediate`]) or
    /// later by the device during collective execution ([`ScalarResidence::Device`]).
    /// # Safety
    /// The pointer must match the residence and remain readable for every native
    /// use. Device scalars must belong to this context and outlive all queued
    /// collectives and graph replay that use the returned operator.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// Also returns an error if NCCL cannot create the pre-multiplied sum operator.
    pub unsafe fn create_custom_pre_mul_sum<T: DataTypeLike>(
        &self,
        scalar: *mut T,
        residence: ScalarResidence,
    ) -> Result<CustomReductionOperator<'_>> {
        self.bind()?;

        // Dynamic operators are arbitrary native enum integers, not Rust enum variants.
        let mut value: i32 = sys::ncclRedOp_t::ncclNumOps as i32;
        // SAFETY: `ncclRedOpCreatePreMulSum` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::ncclRedOpCreatePreMulSum(
                (&raw mut value).cast(),
                scalar.cast(),
                T::DATA_TYPE.into(),
                residence.into(),
                self.as_raw(),
            ))?;
        }

        Ok(CustomReductionOperator {
            communicator: self,
            raw: value,
        })
    }

    /// Creates a sum reduction operator that copies `scalar` from host memory and multiplies each input by it before summation.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// Also returns an error if NCCL cannot create the pre-multiplied sum operator.
    pub fn create_custom_pre_mul_sum_host<T: DataTypeLike>(
        &self,
        scalar: &T,
    ) -> Result<CustomReductionOperator<'_>> {
        let scalar = ptr::from_ref::<T>(scalar).cast_mut();
        // SAFETY: NCCL copies the valid host scalar before returning.
        unsafe { self.create_custom_pre_mul_sum(scalar, ScalarResidence::HostImmediate) }
    }

    /// # Safety
    /// Retain the scalar allocation and order its reads through completion of
    /// every collective and graph replay using this operator.
    ///
    /// # Errors
    ///
    /// Returns an error if `scalar` does not contain exactly one element or belongs to another CUDA context.
    /// Communicator-state checks, CUDA context binding, and NCCL operator-creation errors are propagated.
    pub unsafe fn create_custom_pre_mul_sum_device<T: DataTypeLike>(
        &self,
        scalar: DeviceView<'_, T>,
    ) -> Result<CustomReductionOperator<'_>> {
        if scalar.len() != 1 {
            return Err(Error::LengthMismatch {
                name: "custom reduction scalar buffer".into(),
            });
        }
        self.ensure_buffer_context(scalar.context())?;
        // SAFETY: the caller retains the initialized device scalar through all uses.
        unsafe {
            self.create_custom_pre_mul_sum(scalar.as_ptr().cast_mut(), ScalarResidence::Device)
        }
    }
}

// NCCL dynamic operator values are outside the generated Rust enum's variants.
// Preserve the native C enum ABI as an integer when passing a dynamic handle.
unsafe extern "C" {
    #[link_name = "ncclRedOpDestroy"]
    pub(crate) fn destroy_dynamic_operator(op: i32, comm: sys::ncclComm_t) -> sys::ncclResult_t;
}
