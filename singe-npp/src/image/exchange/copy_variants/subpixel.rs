use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange_validation::validate_subpixel_copy,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::DataTypeLike,
};

/// Copies source samples at fractional pixel shifts `x_shift` and `y_shift` into the destination using linear interpolation.
///
/// Enqueues work on the selected stream.
/// Row padding is not initialized.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
/// Both shifts must be finite and in `0.0..1.0`.
/// Destination width and height must not exceed the corresponding source dimensions.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// The source allocation must include every neighboring sample read by interpolation, including samples outside the declared view when required by the native routine.
/// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn copy_subpixel_u8_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, u8, C1>,
    x_shift: f32,
    y_shift: f32,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;

    validate_subpixel_copy(source.size(), destination.size(), x_shift, y_shift)?;

    // SAFETY: `nppiCopySubpix_8u_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCopySubpix_8u_C1R_Ctx(
            source.as_ptr().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            destination.size().into(),
            x_shift,
            y_shift,
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

macro_rules! impl_copy_subpixel {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        /// Copies source samples at fractional pixel shifts `x_shift` and `y_shift` into the destination using linear interpolation.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Both shifts must be finite and in `0.0..1.0`.
        /// Destination width and height must not exceed the corresponding source dimensions.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// The source allocation must include every neighboring sample read by interpolation, including samples outside the declared view when required by the native routine.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            x_shift: f32,
            y_shift: f32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_subpixel_copy(source.size(), destination.size(), x_shift, y_shift)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    x_shift,
                    y_shift,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_copy_subpixel_operation {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),+ $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and use a compatible subpixel-copy routine.
        /// They must uphold the method's shift and extent checks, interpolation footprint, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Copies source samples at fractional pixel shifts `x_shift` and `y_shift` into the destination using linear interpolation.
            ///
            /// Enqueues work on the selected stream.
            /// For AC4 layouts, the destination alpha channel is not processed.
            /// Row padding is not initialized.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Both shifts must be finite and in `0.0..1.0`.
            /// Destination width and height must not exceed the corresponding source dimensions.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// The source allocation must include every neighboring sample read by interpolation, including samples outside the declared view when required by the native routine.
            /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn dispatch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                x_shift: f32,
                y_shift: f32,
            ) -> Result<()>;
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn dispatch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    x_shift: f32,
                    y_shift: f32,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, x_shift, y_shift)
                    }
}
            }
        )+

        /// Copies source samples at fractional pixel shifts `x_shift` and `y_shift` into the destination using linear interpolation.
        ///
        /// Enqueues work on the selected stream.
        /// For AC4 layouts, the destination alpha channel is not processed.
        /// Row padding is not initialized.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Both shifts must be finite and in `0.0..1.0`.
        /// Destination width and height must not exceed the corresponding source dimensions.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// The source allocation must include every neighboring sample read by interpolation, including samples outside the declared view when required by the native routine.
        /// Keep source storage read-only and destination storage exclusively writable and disjoint from the source during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            x_shift: f32,
            y_shift: f32,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::dispatch(stream_context, source, destination, x_shift, y_shift)
            }
}
    };
}

impl_copy_subpixel!(copy_subpixel_u8_c3, u8, C3, nppiCopySubpix_8u_C3R_Ctx);

impl_copy_subpixel!(copy_subpixel_u8_c4, u8, C4, nppiCopySubpix_8u_C4R_Ctx);

impl_copy_subpixel!(copy_subpixel_u8_ac4, u8, AC4, nppiCopySubpix_8u_AC4R_Ctx);

impl_copy_subpixel!(copy_subpixel_u16_c1, u16, C1, nppiCopySubpix_16u_C1R_Ctx);

impl_copy_subpixel!(copy_subpixel_u16_c3, u16, C3, nppiCopySubpix_16u_C3R_Ctx);

impl_copy_subpixel!(copy_subpixel_u16_c4, u16, C4, nppiCopySubpix_16u_C4R_Ctx);

impl_copy_subpixel!(copy_subpixel_u16_ac4, u16, AC4, nppiCopySubpix_16u_AC4R_Ctx);

impl_copy_subpixel!(copy_subpixel_i16_c1, i16, C1, nppiCopySubpix_16s_C1R_Ctx);

impl_copy_subpixel!(copy_subpixel_i16_c3, i16, C3, nppiCopySubpix_16s_C3R_Ctx);

impl_copy_subpixel!(copy_subpixel_i16_c4, i16, C4, nppiCopySubpix_16s_C4R_Ctx);

impl_copy_subpixel!(copy_subpixel_i16_ac4, i16, AC4, nppiCopySubpix_16s_AC4R_Ctx);

impl_copy_subpixel!(copy_subpixel_i32_c1, i32, C1, nppiCopySubpix_32s_C1R_Ctx);

impl_copy_subpixel!(copy_subpixel_i32_c3, i32, C3, nppiCopySubpix_32s_C3R_Ctx);

impl_copy_subpixel!(copy_subpixel_i32_c4, i32, C4, nppiCopySubpix_32s_C4R_Ctx);

impl_copy_subpixel!(copy_subpixel_i32_ac4, i32, AC4, nppiCopySubpix_32s_AC4R_Ctx);

impl_copy_subpixel!(copy_subpixel_f32_c1, f32, C1, nppiCopySubpix_32f_C1R_Ctx);

impl_copy_subpixel!(copy_subpixel_f32_c3, f32, C3, nppiCopySubpix_32f_C3R_Ctx);

impl_copy_subpixel!(copy_subpixel_f32_c4, f32, C4, nppiCopySubpix_32f_C4R_Ctx);

impl_copy_subpixel!(copy_subpixel_f32_ac4, f32, AC4, nppiCopySubpix_32f_AC4R_Ctx);

impl_generic_copy_subpixel_operation!(
    CopySubpixelC1,
    copy_subpixel_c1,
    C1,
    [
        u8 => copy_subpixel_u8_c1,
        u16 => copy_subpixel_u16_c1,
        i16 => copy_subpixel_i16_c1,
        i32 => copy_subpixel_i32_c1,
        f32 => copy_subpixel_f32_c1,
    ]
);

impl_generic_copy_subpixel_operation!(
    CopySubpixelC3,
    copy_subpixel_c3,
    C3,
    [
        u8 => copy_subpixel_u8_c3,
        u16 => copy_subpixel_u16_c3,
        i16 => copy_subpixel_i16_c3,
        i32 => copy_subpixel_i32_c3,
        f32 => copy_subpixel_f32_c3,
    ]
);

impl_generic_copy_subpixel_operation!(
    CopySubpixelC4,
    copy_subpixel_c4,
    C4,
    [
        u8 => copy_subpixel_u8_c4,
        u16 => copy_subpixel_u16_c4,
        i16 => copy_subpixel_i16_c4,
        i32 => copy_subpixel_i32_c4,
        f32 => copy_subpixel_f32_c4,
    ]
);

impl_generic_copy_subpixel_operation!(
    CopySubpixelAC4,
    copy_subpixel_ac4,
    AC4,
    [
        u8 => copy_subpixel_u8_ac4,
        u16 => copy_subpixel_u16_ac4,
        i16 => copy_subpixel_i16_ac4,
        i32 => copy_subpixel_i32_ac4,
        f32 => copy_subpixel_f32_ac4,
    ]
);
