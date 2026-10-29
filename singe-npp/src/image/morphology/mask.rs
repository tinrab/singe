use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        morphology::{validate_mask, validate_same_size},
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{DataTypeLike, Point, Size},
};

macro_rules! impl_morphology_mask {
    ($doc:literal, $name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Mask storage must satisfy the native structuring-element and address-space requirements and remain readable for as long as native work uses it.
        /// The source allocation must include every neighboring sample read by the structuring element, including any halo outside the declared view.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            mask: &[u8],
            mask_size: Size,
            anchor: Point,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source.size(), destination.size())?;
            validate_mask(mask, mask_size, anchor)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source.size().into(),
                    mask.as_ptr().cast(),
                    mask_size.into(),
                    anchor.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_morphology_mask {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native image and structuring-element ABI and use a compatible morphology routine.
        /// They must uphold the method's geometry checks, access rules, and queued lifetimes.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// Source and destination image sizes must match.
            /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Mask storage must satisfy the native structuring-element and address-space requirements and remain readable for as long as native work uses it.
            /// The source allocation must include every neighboring sample read by the structuring element, including any halo outside the declared view.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
            ) -> Result<()>;
        }

        #[doc = $doc]
        ///
        /// Enqueues work on the selected stream.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// Source and destination image sizes must match.
        /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Mask storage must satisfy the native structuring-element and address-space requirements and remain readable for as long as native work uses it.
        /// The source allocation must include every neighboring sample read by the structuring element, including any halo outside the declared view.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
            mask: &[u8],
            mask_size: Size,
            anchor: Point,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination, mask, mask_size, anchor)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    mask: &[u8],
                    mask_size: Size,
                    anchor: Point,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination, mask, mask_size, anchor)
                    }
}
            }
        )*
    };
}

impl_morphology_mask!(
    "Single-channel 8-bit unsigned integer dilation.",
    dilate_u8_c1,
    u8,
    C1,
    nppiDilate_8u_C1R_Ctx
);

impl_morphology_mask!(
    "Three-channel 8-bit unsigned integer dilation.",
    dilate_u8_c3,
    u8,
    C3,
    nppiDilate_8u_C3R_Ctx
);

impl_morphology_mask!(
    "Four-channel 8-bit unsigned integer dilation.",
    dilate_u8_c4,
    u8,
    C4,
    nppiDilate_8u_C4R_Ctx
);

impl_morphology_mask!(
    "Four-channel 8-bit unsigned integer dilation, ignoring alpha-channel.",
    dilate_u8_ac4,
    u8,
    AC4,
    nppiDilate_8u_AC4R_Ctx
);

impl_morphology_mask!(
    "Single-channel 16-bit unsigned integer dilation.",
    dilate_u16_c1,
    u16,
    C1,
    nppiDilate_16u_C1R_Ctx
);

impl_morphology_mask!(
    "Three-channel 16-bit unsigned integer dilation.",
    dilate_u16_c3,
    u16,
    C3,
    nppiDilate_16u_C3R_Ctx
);

impl_morphology_mask!(
    "Four-channel 16-bit unsigned integer dilation.",
    dilate_u16_c4,
    u16,
    C4,
    nppiDilate_16u_C4R_Ctx
);

impl_morphology_mask!(
    "Four-channel 16-bit unsigned integer dilation, ignoring alpha-channel.",
    dilate_u16_ac4,
    u16,
    AC4,
    nppiDilate_16u_AC4R_Ctx
);

impl_morphology_mask!(
    "Single-channel 32-bit floating-point dilation.",
    dilate_f32_c1,
    f32,
    C1,
    nppiDilate_32f_C1R_Ctx
);

impl_morphology_mask!(
    "Three-channel 32-bit floating-point dilation.",
    dilate_f32_c3,
    f32,
    C3,
    nppiDilate_32f_C3R_Ctx
);

impl_morphology_mask!(
    "Four-channel 32-bit floating-point dilation.",
    dilate_f32_c4,
    f32,
    C4,
    nppiDilate_32f_C4R_Ctx
);

impl_morphology_mask!(
    "Four-channel 32-bit floating-point dilation, ignoring alpha-channel.",
    dilate_f32_ac4,
    f32,
    AC4,
    nppiDilate_32f_AC4R_Ctx
);

impl_morphology_mask!(
    "Single-channel 8-bit unsigned integer erosion.",
    erode_u8_c1,
    u8,
    C1,
    nppiErode_8u_C1R_Ctx
);

impl_morphology_mask!(
    "Three-channel 8-bit unsigned integer erosion.",
    erode_u8_c3,
    u8,
    C3,
    nppiErode_8u_C3R_Ctx
);

impl_morphology_mask!(
    "Four-channel 8-bit unsigned integer erosion.",
    erode_u8_c4,
    u8,
    C4,
    nppiErode_8u_C4R_Ctx
);

impl_morphology_mask!(
    "Four-channel 8-bit unsigned integer erosion, ignoring alpha-channel.",
    erode_u8_ac4,
    u8,
    AC4,
    nppiErode_8u_AC4R_Ctx
);

impl_morphology_mask!(
    "Single-channel 16-bit unsigned integer erosion.",
    erode_u16_c1,
    u16,
    C1,
    nppiErode_16u_C1R_Ctx
);

impl_morphology_mask!(
    "Three-channel 16-bit unsigned integer erosion.",
    erode_u16_c3,
    u16,
    C3,
    nppiErode_16u_C3R_Ctx
);

impl_morphology_mask!(
    "Four-channel 16-bit unsigned integer erosion.",
    erode_u16_c4,
    u16,
    C4,
    nppiErode_16u_C4R_Ctx
);

impl_morphology_mask!(
    "Four-channel 16-bit unsigned integer erosion, ignoring alpha-channel.",
    erode_u16_ac4,
    u16,
    AC4,
    nppiErode_16u_AC4R_Ctx
);

impl_morphology_mask!(
    "Single-channel 32-bit floating-point erosion.",
    erode_f32_c1,
    f32,
    C1,
    nppiErode_32f_C1R_Ctx
);

impl_morphology_mask!(
    "Three-channel 32-bit floating-point erosion.",
    erode_f32_c3,
    f32,
    C3,
    nppiErode_32f_C3R_Ctx
);

impl_morphology_mask!(
    "Four-channel 32-bit floating-point erosion.",
    erode_f32_c4,
    f32,
    C4,
    nppiErode_32f_C4R_Ctx
);

impl_morphology_mask!(
    "Four-channel 32-bit floating-point erosion, ignoring alpha-channel.",
    erode_f32_ac4,
    f32,
    AC4,
    nppiErode_32f_AC4R_Ctx
);

impl_generic_morphology_mask!(
    "Performs dilation using `mask`, `mask_size`, and `anchor`.",
DilateC1, dilate, dilate_c1, C1, [
    u8 => dilate_u8_c1,
    u16 => dilate_u16_c1,
    f32 => dilate_f32_c1,
]
);

impl_generic_morphology_mask!(
    "Performs dilation using `mask`, `mask_size`, and `anchor`.",
DilateC3, dilate, dilate_c3, C3, [
    u8 => dilate_u8_c3,
    u16 => dilate_u16_c3,
    f32 => dilate_f32_c3,
]
);

impl_generic_morphology_mask!(
    "Performs dilation using `mask`, `mask_size`, and `anchor`.",
DilateC4, dilate, dilate_c4, C4, [
    u8 => dilate_u8_c4,
    u16 => dilate_u16_c4,
    f32 => dilate_f32_c4,
]
);

impl_generic_morphology_mask!(
    "Performs dilation using `mask`, `mask_size`, and `anchor`. The destination alpha channel is not processed.",
DilateAc4, dilate, dilate_ac4, AC4, [
    u8 => dilate_u8_ac4,
    u16 => dilate_u16_ac4,
    f32 => dilate_f32_ac4,
]
);

impl_generic_morphology_mask!(
    "Performs erosion using `mask`, `mask_size`, and `anchor`.",
ErodeC1, erode, erode_c1, C1, [
    u8 => erode_u8_c1,
    u16 => erode_u16_c1,
    f32 => erode_f32_c1,
]
);

impl_generic_morphology_mask!(
    "Performs erosion using `mask`, `mask_size`, and `anchor`.",
ErodeC3, erode, erode_c3, C3, [
    u8 => erode_u8_c3,
    u16 => erode_u16_c3,
    f32 => erode_f32_c3,
]
);

impl_generic_morphology_mask!(
    "Performs erosion using `mask`, `mask_size`, and `anchor`.",
ErodeC4, erode, erode_c4, C4, [
    u8 => erode_u8_c4,
    u16 => erode_u16_c4,
    f32 => erode_f32_c4,
]
);

impl_generic_morphology_mask!(
    "Performs erosion using `mask`, `mask_size`, and `anchor`. The destination alpha channel is not processed.",
ErodeAc4, erode, erode_ac4, AC4, [
    u8 => erode_u8_ac4,
    u16 => erode_u16_ac4,
    f32 => erode_f32_ac4,
]
);
