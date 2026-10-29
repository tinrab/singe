use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        morphology::validate_border_roi,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point},
};

macro_rules! impl_morphology_3x3_border {
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
        /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            border_type: BorderType,
        ) -> Result<()> {
            // NPP border offsets describe the position already addressed by pSrc within the full image.
            let source_pointer = source
                .window($crate::types::Rectangle {
                    x: source_offset.x,
                    y: source_offset.y,
                    width: 1,
                    height: 1,
                })?
                .view()
                .as_ptr();

            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(destination)?;

            validate_border_roi(source.size(), source_offset, destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_pointer.cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    destination.size().into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_morphology_3x3_border {
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
            /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                border_type: BorderType,
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
        /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, T, $layout>,
            border_type: BorderType,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_offset: Point,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    border_type: BorderType,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            source_offset,
                            destination,
                            border_type,
                        )
                    }
                }
            }
        )*
    };
}

impl_morphology_3x3_border!(
    "Single-channel 8-bit unsigned integer 3x3 dilation with border control.",
    dilate_3x3_border_u8_c1,
    u8,
    C1,
    nppiDilate3x3Border_8u_C1R_Ctx
);

impl_morphology_3x3_border!(
    "Three-channel 8-bit unsigned integer 3x3 dilation with border control.",
    dilate_3x3_border_u8_c3,
    u8,
    C3,
    nppiDilate3x3Border_8u_C3R_Ctx
);

impl_morphology_3x3_border!(
    "Four-channel 8-bit unsigned integer 3x3 dilation with border control.",
    dilate_3x3_border_u8_c4,
    u8,
    C4,
    nppiDilate3x3Border_8u_C4R_Ctx
);

impl_morphology_3x3_border!(
    "Four-channel 8-bit unsigned integer 3x3 dilation with border control, ignoring alpha-channel.",
    dilate_3x3_border_u8_ac4,
    u8,
    AC4,
    nppiDilate3x3Border_8u_AC4R_Ctx
);

impl_morphology_3x3_border!(
    "Single-channel 16-bit unsigned integer 3x3 dilation with border control.",
    dilate_3x3_border_u16_c1,
    u16,
    C1,
    nppiDilate3x3Border_16u_C1R_Ctx
);

impl_morphology_3x3_border!(
    "Three-channel 16-bit unsigned integer 3x3 dilation with border control.",
    dilate_3x3_border_u16_c3,
    u16,
    C3,
    nppiDilate3x3Border_16u_C3R_Ctx
);

impl_morphology_3x3_border!(
    "Four-channel 16-bit unsigned integer 3x3 dilation with border control.",
    dilate_3x3_border_u16_c4,
    u16,
    C4,
    nppiDilate3x3Border_16u_C4R_Ctx
);

impl_morphology_3x3_border!(
    "Four-channel 16-bit unsigned integer 3x3 dilation with border control, ignoring alpha-channel.",
    dilate_3x3_border_u16_ac4,
    u16,
    AC4,
    nppiDilate3x3Border_16u_AC4R_Ctx
);

impl_morphology_3x3_border!(
    "Single-channel 32-bit floating-point 3x3 dilation with border control.",
    dilate_3x3_border_f32_c1,
    f32,
    C1,
    nppiDilate3x3Border_32f_C1R_Ctx
);

impl_morphology_3x3_border!(
    "Three-channel 32-bit floating-point 3x3 dilation with border control.",
    dilate_3x3_border_f32_c3,
    f32,
    C3,
    nppiDilate3x3Border_32f_C3R_Ctx
);

impl_morphology_3x3_border!(
    "Four-channel 32-bit floating-point 3x3 dilation with border control.",
    dilate_3x3_border_f32_c4,
    f32,
    C4,
    nppiDilate3x3Border_32f_C4R_Ctx
);

impl_morphology_3x3_border!(
    "Four-channel 32-bit floating-point 3x3 dilation with border control, ignoring alpha-channel.",
    dilate_3x3_border_f32_ac4,
    f32,
    AC4,
    nppiDilate3x3Border_32f_AC4R_Ctx
);

impl_morphology_3x3_border!(
    "Single-channel 8-bit unsigned integer 3x3 erosion with border control.",
    erode_3x3_border_u8_c1,
    u8,
    C1,
    nppiErode3x3Border_8u_C1R_Ctx
);

impl_morphology_3x3_border!(
    "Three-channel 8-bit unsigned integer 3x3 erosion with border control.",
    erode_3x3_border_u8_c3,
    u8,
    C3,
    nppiErode3x3Border_8u_C3R_Ctx
);

impl_morphology_3x3_border!(
    "Four-channel 8-bit unsigned integer 3x3 erosion with border control.",
    erode_3x3_border_u8_c4,
    u8,
    C4,
    nppiErode3x3Border_8u_C4R_Ctx
);

impl_morphology_3x3_border!(
    "Four-channel 8-bit unsigned integer 3x3 erosion with border control, ignoring alpha-channel.",
    erode_3x3_border_u8_ac4,
    u8,
    AC4,
    nppiErode3x3Border_8u_AC4R_Ctx
);

impl_morphology_3x3_border!(
    "Single-channel 16-bit unsigned integer 3x3 erosion with border control.",
    erode_3x3_border_u16_c1,
    u16,
    C1,
    nppiErode3x3Border_16u_C1R_Ctx
);

impl_morphology_3x3_border!(
    "Three-channel 16-bit unsigned integer 3x3 erosion with border control.",
    erode_3x3_border_u16_c3,
    u16,
    C3,
    nppiErode3x3Border_16u_C3R_Ctx
);

impl_morphology_3x3_border!(
    "Four-channel 16-bit unsigned integer 3x3 erosion with border control.",
    erode_3x3_border_u16_c4,
    u16,
    C4,
    nppiErode3x3Border_16u_C4R_Ctx
);

impl_morphology_3x3_border!(
    "Four-channel 16-bit unsigned integer 3x3 erosion with border control, ignoring alpha-channel.",
    erode_3x3_border_u16_ac4,
    u16,
    AC4,
    nppiErode3x3Border_16u_AC4R_Ctx
);

impl_morphology_3x3_border!(
    "Single-channel 32-bit floating-point 3x3 erosion with border control.",
    erode_3x3_border_f32_c1,
    f32,
    C1,
    nppiErode3x3Border_32f_C1R_Ctx
);

impl_morphology_3x3_border!(
    "Three-channel 32-bit floating-point 3x3 erosion with border control.",
    erode_3x3_border_f32_c3,
    f32,
    C3,
    nppiErode3x3Border_32f_C3R_Ctx
);

impl_morphology_3x3_border!(
    "Four-channel 32-bit floating-point 3x3 erosion with border control.",
    erode_3x3_border_f32_c4,
    f32,
    C4,
    nppiErode3x3Border_32f_C4R_Ctx
);

impl_morphology_3x3_border!(
    "Four-channel 32-bit floating-point 3x3 erosion with border control, ignoring alpha-channel.",
    erode_3x3_border_f32_ac4,
    f32,
    AC4,
    nppiErode3x3Border_32f_AC4R_Ctx
);

impl_generic_morphology_3x3_border!(
    "Performs dilation using a 3 by 3 neighborhood with border control.",

    Dilate3x3BorderC1,
    dilate_3x3_border,
    dilate_3x3_border_c1,
    C1,
    [
        u8 => dilate_3x3_border_u8_c1,
        u16 => dilate_3x3_border_u16_c1,
        f32 => dilate_3x3_border_f32_c1,
    ]

);

impl_generic_morphology_3x3_border!(
    "Performs dilation using a 3 by 3 neighborhood with border control.",

    Dilate3x3BorderC3,
    dilate_3x3_border,
    dilate_3x3_border_c3,
    C3,
    [
        u8 => dilate_3x3_border_u8_c3,
        u16 => dilate_3x3_border_u16_c3,
        f32 => dilate_3x3_border_f32_c3,
    ]

);

impl_generic_morphology_3x3_border!(
    "Performs dilation using a 3 by 3 neighborhood with border control.",

    Dilate3x3BorderC4,
    dilate_3x3_border,
    dilate_3x3_border_c4,
    C4,
    [
        u8 => dilate_3x3_border_u8_c4,
        u16 => dilate_3x3_border_u16_c4,
        f32 => dilate_3x3_border_f32_c4,
    ]

);

impl_generic_morphology_3x3_border!(
    "Performs dilation using a 3 by 3 neighborhood with border control. The destination alpha channel is not processed.",

    Dilate3x3BorderAc4,
    dilate_3x3_border,
    dilate_3x3_border_ac4,
    AC4,
    [
        u8 => dilate_3x3_border_u8_ac4,
        u16 => dilate_3x3_border_u16_ac4,
        f32 => dilate_3x3_border_f32_ac4,
    ]

);

impl_generic_morphology_3x3_border!(
    "Performs erosion using a 3 by 3 neighborhood with border control.",

    Erode3x3BorderC1,
    erode_3x3_border,
    erode_3x3_border_c1,
    C1,
    [
        u8 => erode_3x3_border_u8_c1,
        u16 => erode_3x3_border_u16_c1,
        f32 => erode_3x3_border_f32_c1,
    ]

);

impl_generic_morphology_3x3_border!(
    "Performs erosion using a 3 by 3 neighborhood with border control.",

    Erode3x3BorderC3,
    erode_3x3_border,
    erode_3x3_border_c3,
    C3,
    [
        u8 => erode_3x3_border_u8_c3,
        u16 => erode_3x3_border_u16_c3,
        f32 => erode_3x3_border_f32_c3,
    ]

);

impl_generic_morphology_3x3_border!(
    "Performs erosion using a 3 by 3 neighborhood with border control.",

    Erode3x3BorderC4,
    erode_3x3_border,
    erode_3x3_border_c4,
    C4,
    [
        u8 => erode_3x3_border_u8_c4,
        u16 => erode_3x3_border_u16_c4,
        f32 => erode_3x3_border_f32_c4,
    ]

);

impl_generic_morphology_3x3_border!(
    "Performs erosion using a 3 by 3 neighborhood with border control. The destination alpha channel is not processed.",

    Erode3x3BorderAc4,
    erode_3x3_border,
    erode_3x3_border_ac4,
    AC4,
    [
        u8 => erode_3x3_border_u8_ac4,
        u16 => erode_3x3_border_u16_ac4,
        f32 => erode_3x3_border_f32_ac4,
    ]

);
