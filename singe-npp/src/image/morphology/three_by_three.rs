use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        morphology::validate_same_size,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::DataTypeLike,
};

macro_rules! impl_morphology_3x3 {
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
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// The source allocation must include every neighboring sample read by the structuring element, including any halo outside the declared view.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
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
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_morphology_3x3 {
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
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// The source allocation must include every neighboring sample read by the structuring element, including any halo outside the declared view.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                destination: &mut ImageOutputView<'_, Self, Layout>,
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
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// The source allocation must include every neighboring sample read by the structuring element, including any halo outside the declared view.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            destination: &mut ImageOutputView<'_, T, $layout>,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(stream_context, source, destination)
            }
}

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(stream_context, source, destination)
                    }
}
            }
        )*
    };
}

impl_morphology_3x3!(
    "Single-channel 8-bit unsigned integer 3x3 dilation.",
    dilate_3x3_u8_c1,
    u8,
    C1,
    nppiDilate3x3_8u_C1R_Ctx
);

impl_morphology_3x3!(
    "Three-channel 8-bit unsigned integer 3x3 dilation.",
    dilate_3x3_u8_c3,
    u8,
    C3,
    nppiDilate3x3_8u_C3R_Ctx
);

impl_morphology_3x3!(
    "Four-channel 8-bit unsigned integer 3x3 dilation.",
    dilate_3x3_u8_c4,
    u8,
    C4,
    nppiDilate3x3_8u_C4R_Ctx
);

impl_morphology_3x3!(
    "Four-channel 8-bit unsigned integer 3x3 dilation, ignoring alpha-channel.",
    dilate_3x3_u8_ac4,
    u8,
    AC4,
    nppiDilate3x3_8u_AC4R_Ctx
);

impl_morphology_3x3!(
    "Single-channel 16-bit unsigned integer 3x3 dilation.",
    dilate_3x3_u16_c1,
    u16,
    C1,
    nppiDilate3x3_16u_C1R_Ctx
);

impl_morphology_3x3!(
    "Three-channel 16-bit unsigned integer 3x3 dilation.",
    dilate_3x3_u16_c3,
    u16,
    C3,
    nppiDilate3x3_16u_C3R_Ctx
);

impl_morphology_3x3!(
    "Four-channel 16-bit unsigned integer 3x3 dilation.",
    dilate_3x3_u16_c4,
    u16,
    C4,
    nppiDilate3x3_16u_C4R_Ctx
);

impl_morphology_3x3!(
    "Four-channel 16-bit unsigned integer 3x3 dilation, ignoring alpha-channel.",
    dilate_3x3_u16_ac4,
    u16,
    AC4,
    nppiDilate3x3_16u_AC4R_Ctx
);

impl_morphology_3x3!(
    "Single-channel 32-bit floating-point 3x3 dilation.",
    dilate_3x3_f32_c1,
    f32,
    C1,
    nppiDilate3x3_32f_C1R_Ctx
);

impl_morphology_3x3!(
    "Three-channel 32-bit floating-point 3x3 dilation.",
    dilate_3x3_f32_c3,
    f32,
    C3,
    nppiDilate3x3_32f_C3R_Ctx
);

impl_morphology_3x3!(
    "Four-channel 32-bit floating-point 3x3 dilation.",
    dilate_3x3_f32_c4,
    f32,
    C4,
    nppiDilate3x3_32f_C4R_Ctx
);

impl_morphology_3x3!(
    "Four-channel 32-bit floating-point 3x3 dilation, ignoring alpha-channel.",
    dilate_3x3_f32_ac4,
    f32,
    AC4,
    nppiDilate3x3_32f_AC4R_Ctx
);

impl_morphology_3x3!(
    "Single-channel 64-bit floating-point 3x3 dilation.",
    dilate_3x3_f64_c1,
    f64,
    C1,
    nppiDilate3x3_64f_C1R_Ctx
);

impl_morphology_3x3!(
    "Single-channel 8-bit unsigned integer 3x3 erosion.",
    erode_3x3_u8_c1,
    u8,
    C1,
    nppiErode3x3_8u_C1R_Ctx
);

impl_morphology_3x3!(
    "Three-channel 8-bit unsigned integer 3x3 erosion.",
    erode_3x3_u8_c3,
    u8,
    C3,
    nppiErode3x3_8u_C3R_Ctx
);

impl_morphology_3x3!(
    "Four-channel 8-bit unsigned integer 3x3 erosion.",
    erode_3x3_u8_c4,
    u8,
    C4,
    nppiErode3x3_8u_C4R_Ctx
);

impl_morphology_3x3!(
    "Four-channel 8-bit unsigned integer 3x3 erosion, ignoring alpha-channel.",
    erode_3x3_u8_ac4,
    u8,
    AC4,
    nppiErode3x3_8u_AC4R_Ctx
);

impl_morphology_3x3!(
    "Single-channel 16-bit unsigned integer 3x3 erosion.",
    erode_3x3_u16_c1,
    u16,
    C1,
    nppiErode3x3_16u_C1R_Ctx
);

impl_morphology_3x3!(
    "Three-channel 16-bit unsigned integer 3x3 erosion.",
    erode_3x3_u16_c3,
    u16,
    C3,
    nppiErode3x3_16u_C3R_Ctx
);

impl_morphology_3x3!(
    "Four-channel 16-bit unsigned integer 3x3 erosion.",
    erode_3x3_u16_c4,
    u16,
    C4,
    nppiErode3x3_16u_C4R_Ctx
);

impl_morphology_3x3!(
    "Four-channel 16-bit unsigned integer 3x3 erosion, ignoring alpha-channel.",
    erode_3x3_u16_ac4,
    u16,
    AC4,
    nppiErode3x3_16u_AC4R_Ctx
);

impl_morphology_3x3!(
    "Single-channel 32-bit floating-point 3x3 erosion.",
    erode_3x3_f32_c1,
    f32,
    C1,
    nppiErode3x3_32f_C1R_Ctx
);

impl_morphology_3x3!(
    "Three-channel 32-bit floating-point 3x3 erosion.",
    erode_3x3_f32_c3,
    f32,
    C3,
    nppiErode3x3_32f_C3R_Ctx
);

impl_morphology_3x3!(
    "Four-channel 32-bit floating-point 3x3 erosion.",
    erode_3x3_f32_c4,
    f32,
    C4,
    nppiErode3x3_32f_C4R_Ctx
);

impl_morphology_3x3!(
    "Four-channel 32-bit floating-point 3x3 erosion, ignoring alpha-channel.",
    erode_3x3_f32_ac4,
    f32,
    AC4,
    nppiErode3x3_32f_AC4R_Ctx
);

impl_morphology_3x3!(
    "Single-channel 64-bit floating-point 3x3 erosion.",
    erode_3x3_f64_c1,
    f64,
    C1,
    nppiErode3x3_64f_C1R_Ctx
);

impl_generic_morphology_3x3!(
    "Performs dilation using a 3 by 3 neighborhood.",
Dilate3x3C1, dilate_3x3, dilate_3x3_c1, C1, [
    u8 => dilate_3x3_u8_c1,
    u16 => dilate_3x3_u16_c1,
    f32 => dilate_3x3_f32_c1,
    f64 => dilate_3x3_f64_c1,
]
);

impl_generic_morphology_3x3!(
    "Performs dilation using a 3 by 3 neighborhood.",
Dilate3x3C3, dilate_3x3, dilate_3x3_c3, C3, [
    u8 => dilate_3x3_u8_c3,
    u16 => dilate_3x3_u16_c3,
    f32 => dilate_3x3_f32_c3,
]
);

impl_generic_morphology_3x3!(
    "Performs dilation using a 3 by 3 neighborhood.",
Dilate3x3C4, dilate_3x3, dilate_3x3_c4, C4, [
    u8 => dilate_3x3_u8_c4,
    u16 => dilate_3x3_u16_c4,
    f32 => dilate_3x3_f32_c4,
]
);

impl_generic_morphology_3x3!(
    "Performs dilation using a 3 by 3 neighborhood. The destination alpha channel is not processed.",
Dilate3x3Ac4, dilate_3x3, dilate_3x3_ac4, AC4, [
    u8 => dilate_3x3_u8_ac4,
    u16 => dilate_3x3_u16_ac4,
    f32 => dilate_3x3_f32_ac4,
]
);

impl_generic_morphology_3x3!(
    "Performs erosion using a 3 by 3 neighborhood.",
Erode3x3C1, erode_3x3, erode_3x3_c1, C1, [
    u8 => erode_3x3_u8_c1,
    u16 => erode_3x3_u16_c1,
    f32 => erode_3x3_f32_c1,
    f64 => erode_3x3_f64_c1,
]
);

impl_generic_morphology_3x3!(
    "Performs erosion using a 3 by 3 neighborhood.",
Erode3x3C3, erode_3x3, erode_3x3_c3, C3, [
    u8 => erode_3x3_u8_c3,
    u16 => erode_3x3_u16_c3,
    f32 => erode_3x3_f32_c3,
]
);

impl_generic_morphology_3x3!(
    "Performs erosion using a 3 by 3 neighborhood.",
Erode3x3C4, erode_3x3, erode_3x3_c4, C4, [
    u8 => erode_3x3_u8_c4,
    u16 => erode_3x3_u16_c4,
    f32 => erode_3x3_f32_c4,
]
);

impl_generic_morphology_3x3!(
    "Performs erosion using a 3 by 3 neighborhood. The destination alpha channel is not processed.",
Erode3x3Ac4, erode_3x3, erode_3x3_ac4, AC4, [
    u8 => erode_3x3_u8_ac4,
    u16 => erode_3x3_u16_ac4,
    f32 => erode_3x3_f32_ac4,
]
);
