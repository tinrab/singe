use crate::{
    buffer::BufferMut,
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
    pub fn create_custom_pre_mul_sum<T: DataTypeLike>(
        &self,
        scalar: *mut T,
        residence: ScalarResidence,
    ) -> Result<CustomReductionOperator<'_>> {
        self.bind()?;

        let mut value = sys::ncclRedOp_t::ncclNumOps;
        unsafe {
            try_ffi!(sys::ncclRedOpCreatePreMulSum(
                &raw mut value,
                scalar.cast(),
                T::nccl_data_type().into(),
                residence.into(),
                self.as_raw(),
            ))?;
        }

        Ok(CustomReductionOperator {
            communicator: self,
            raw: value,
        })
    }

    pub fn create_custom_pre_mul_sum_host<T: DataTypeLike>(
        &self,
        scalar: &T,
    ) -> Result<CustomReductionOperator<'_>> {
        let scalar = (scalar as *const T).cast_mut();
        self.create_custom_pre_mul_sum(scalar, ScalarResidence::HostImmediate)
    }

    pub fn create_custom_pre_mul_sum_device<T: DataTypeLike>(
        &self,
        mut scalar: BufferMut<'_, T>,
    ) -> Result<CustomReductionOperator<'_>> {
        if scalar.len() != 1 {
            return Err(Error::LengthMismatch {
                name: "custom reduction scalar buffer".into(),
            });
        }
        self.create_custom_pre_mul_sum(scalar.as_mut_ptr(), ScalarResidence::Device)
    }
}
