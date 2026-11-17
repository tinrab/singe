use std::{marker::PhantomData, ptr};

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
    /// Pass the returned operator to collectives of `T` launched on this communicator.
    /// `residence` indicates whether `scalar` is dereferenced immediately by the host before this method returns ([`ScalarResidence::HostImmediate`]) or later by the device during collective execution ([`ScalarResidence::Device`]).
    ///
    /// # Safety
    ///
    /// The pointer must match the residence and remain readable for every native use.
    /// Device scalars must belong to this context and outlive all queued collectives and graph replay that use the returned operator.
    ///
    /// # Errors
    ///
    /// Returns an error if the communicator state disallows control operations or binding its CUDA context fails.
    /// Also returns an error if NCCL cannot create the pre-multiplied sum operator.
    pub unsafe fn create_custom_pre_mul_sum<T: DataTypeLike>(
        &self,
        scalar: *mut T,
        residence: ScalarResidence,
    ) -> Result<CustomReductionOperator<'_, T>> {
        self.bind()?;

        let mut value = sys::ncclRedOp_t::ncclNumOps;
        // SAFETY: `self` owns the live communicator with its context bound, `value` is a local output slot, and the caller guarantees `scalar` matches `residence` and stays readable for every use.
        unsafe {
            try_ffi!(sys::ncclRedOpCreatePreMulSum(
                &raw mut value,
                scalar.cast(),
                T::DATA_TYPE.into(),
                residence.into(),
                self.as_raw(),
            ))?;
        }

        Ok(CustomReductionOperator {
            communicator: self,
            raw: value,
            _element: PhantomData,
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
    ) -> Result<CustomReductionOperator<'_, T>> {
        let scalar = ptr::from_ref::<T>(scalar).cast_mut();
        // SAFETY: `scalar` borrows a valid host `T` for this call, and `HostImmediate` makes NCCL copy it before returning.
        unsafe { self.create_custom_pre_mul_sum(scalar, ScalarResidence::HostImmediate) }
    }

    /// Creates a sum reduction operator that multiplies each input by the device scalar in `scalar` before summation.
    ///
    /// # Safety
    ///
    /// Retain the scalar allocation and order its reads through completion of every collective and graph replay using this operator.
    ///
    /// # Errors
    ///
    /// Returns an error if `scalar` does not contain exactly one element or belongs to another CUDA context.
    /// Communicator-state checks, CUDA context binding, and NCCL operator-creation errors are propagated.
    pub unsafe fn create_custom_pre_mul_sum_device<T: DataTypeLike>(
        &self,
        scalar: DeviceView<'_, T>,
    ) -> Result<CustomReductionOperator<'_, T>> {
        if scalar.len() != 1 {
            return Err(Error::LengthMismatch {
                name: "custom reduction scalar buffer".into(),
                expected: 1,
                actual: scalar.len(),
            });
        }

        self.ensure_buffer_context(scalar.context())?;
        // SAFETY: `scalar` is one element in this communicator's context, and the caller guarantees it stays allocated and ordered through every use of the operator.
        unsafe {
            self.create_custom_pre_mul_sum(scalar.as_ptr().cast_mut(), ScalarResidence::Device)
        }
    }
}
