use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic_validation::{device_constant_mut_ptr, validate_same_size},
        view::{C1, ImageOutputView, ImageView},
    },
    try_ffi,
};

macro_rules! impl_absolute_difference_constant {
    ($name:ident, $ty:ty, $ffi:ident) => {
        /// Writes the absolute difference between each `source` pixel and the
        /// scalar `constant` to `destination`.
        ///
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// All image sizes must match; differing sizes return an error.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint
        /// from every input through completion.
        /// Addresses, row strides, and image regions must satisfy the native routine's alignment
        /// and capacity requirements.
        /// Retain images and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            constant: $ty,
            destination: &mut ImageOutputView<'_, $ty, C1>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    constant.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_absolute_difference_device_constant {
    ($name:ident, $ty:ty, $ffi:ident) => {
        /// Writes the absolute difference between each `source` pixel and the
        /// first value in the device `constant` buffer to `destination`.
        ///
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, an image or the constant buffer belongs
        /// to another CUDA context, or the constant buffer contains no element.
        /// All image sizes must match; differing sizes return an error.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// The device constant must also be ready and remain read-only through completion.
        /// Keep source images read-only and destination pixels exclusively writable and disjoint
        /// from every input through completion.
        /// Addresses, row strides, and image regions must satisfy the native routine's alignment
        /// and capacity requirements.
        /// Retain images, device constants, and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            constant: &DeviceMemory<$ty>,
            destination: &mut ImageOutputView<'_, $ty, C1>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(constant)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            let constant = device_constant_mut_ptr(constant, 1)?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    constant.cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_absolute_difference_constant!(
    absolute_difference_constant_u8_c1,
    u8,
    nppiAbsDiffC_8u_C1R_Ctx
);

impl_absolute_difference_constant!(
    absolute_difference_constant_u16_c1,
    u16,
    nppiAbsDiffC_16u_C1R_Ctx
);

impl_absolute_difference_constant!(
    absolute_difference_constant_f32_c1,
    f32,
    nppiAbsDiffC_32f_C1R_Ctx
);

impl_absolute_difference_device_constant!(
    absolute_difference_device_constant_u8_c1,
    u8,
    nppiAbsDiffDeviceC_8u_C1R_Ctx
);

impl_absolute_difference_device_constant!(
    absolute_difference_device_constant_u16_c1,
    u16,
    nppiAbsDiffDeviceC_16u_C1R_Ctx
);

impl_absolute_difference_device_constant!(
    absolute_difference_device_constant_f32_c1,
    f32,
    nppiAbsDiffDeviceC_32f_C1R_Ctx
);
