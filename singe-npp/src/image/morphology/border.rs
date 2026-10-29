use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        morphology::{validate_border_roi, validate_mask, validate_mask_shape},
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point, Size},
};

macro_rules! impl_morphology_mask_border {
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
        /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Mask storage must satisfy the native structuring-element and address-space requirements and remain readable for as long as native work uses it.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            mask: &[u8],
            mask_size: Size,
            anchor: Point,
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
            validate_mask(mask, mask_size, anchor)?;

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
                    mask.as_ptr().cast(),
                    mask_size.into(),
                    anchor.into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_morphology_mask_border {
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
            /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Mask storage must satisfy the native structuring-element and address-space requirements and remain readable for as long as native work uses it.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
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
        /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Mask storage must satisfy the native structuring-element and address-space requirements and remain readable for as long as native work uses it.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, T, $layout>,
            mask: &[u8],
            mask_size: Size,
            anchor: Point,
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
                    mask,
                    mask_size,
                    anchor,
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
                    mask: &[u8],
                    mask_size: Size,
                    anchor: Point,
                    border_type: BorderType,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            source_offset,
                            destination,
                            mask,
                            mask_size,
                            anchor,
                            border_type,
                        )
                    }
                }
            }
        )*
    };
}

impl_morphology_mask_border!(
    "Single-channel 8-bit unsigned integer dilation with border control.",
    dilate_border_u8_c1,
    u8,
    C1,
    nppiDilateBorder_8u_C1R_Ctx
);

impl_morphology_mask_border!(
    "Three-channel 8-bit unsigned integer dilation with border control.",
    dilate_border_u8_c3,
    u8,
    C3,
    nppiDilateBorder_8u_C3R_Ctx
);

impl_morphology_mask_border!(
    "Four-channel 8-bit unsigned integer dilation with border control.",
    dilate_border_u8_c4,
    u8,
    C4,
    nppiDilateBorder_8u_C4R_Ctx
);

impl_morphology_mask_border!(
    "Four-channel 8-bit unsigned integer dilation with border control, ignoring alpha-channel.",
    dilate_border_u8_ac4,
    u8,
    AC4,
    nppiDilateBorder_8u_AC4R_Ctx
);

impl_morphology_mask_border!(
    "Single-channel 16-bit unsigned integer dilation with border control.",
    dilate_border_u16_c1,
    u16,
    C1,
    nppiDilateBorder_16u_C1R_Ctx
);

impl_morphology_mask_border!(
    "Three-channel 16-bit unsigned integer dilation with border control.",
    dilate_border_u16_c3,
    u16,
    C3,
    nppiDilateBorder_16u_C3R_Ctx
);

impl_morphology_mask_border!(
    "Four-channel 16-bit unsigned integer dilation with border control.",
    dilate_border_u16_c4,
    u16,
    C4,
    nppiDilateBorder_16u_C4R_Ctx
);

impl_morphology_mask_border!(
    "Four-channel 16-bit unsigned integer dilation with border control, ignoring alpha-channel.",
    dilate_border_u16_ac4,
    u16,
    AC4,
    nppiDilateBorder_16u_AC4R_Ctx
);

impl_morphology_mask_border!(
    "Single-channel 32-bit floating-point dilation with border control.",
    dilate_border_f32_c1,
    f32,
    C1,
    nppiDilateBorder_32f_C1R_Ctx
);

impl_morphology_mask_border!(
    "Three-channel 32-bit floating-point dilation with border control.",
    dilate_border_f32_c3,
    f32,
    C3,
    nppiDilateBorder_32f_C3R_Ctx
);

impl_morphology_mask_border!(
    "Four-channel 32-bit floating-point dilation with border control.",
    dilate_border_f32_c4,
    f32,
    C4,
    nppiDilateBorder_32f_C4R_Ctx
);

impl_morphology_mask_border!(
    "Four-channel 32-bit floating-point dilation with border control, ignoring alpha-channel.",
    dilate_border_f32_ac4,
    f32,
    AC4,
    nppiDilateBorder_32f_AC4R_Ctx
);

impl_morphology_mask_border!(
    "Single-channel 8-bit unsigned integer erosion with border control.",
    erode_border_u8_c1,
    u8,
    C1,
    nppiErodeBorder_8u_C1R_Ctx
);

impl_morphology_mask_border!(
    "Three-channel 8-bit unsigned integer erosion with border control.",
    erode_border_u8_c3,
    u8,
    C3,
    nppiErodeBorder_8u_C3R_Ctx
);

impl_morphology_mask_border!(
    "Four-channel 8-bit unsigned integer erosion with border control.",
    erode_border_u8_c4,
    u8,
    C4,
    nppiErodeBorder_8u_C4R_Ctx
);

impl_morphology_mask_border!(
    "Four-channel 8-bit unsigned integer erosion with border control, ignoring alpha-channel.",
    erode_border_u8_ac4,
    u8,
    AC4,
    nppiErodeBorder_8u_AC4R_Ctx
);

impl_morphology_mask_border!(
    "Single-channel 16-bit unsigned integer erosion with border control.",
    erode_border_u16_c1,
    u16,
    C1,
    nppiErodeBorder_16u_C1R_Ctx
);

impl_morphology_mask_border!(
    "Three-channel 16-bit unsigned integer erosion with border control.",
    erode_border_u16_c3,
    u16,
    C3,
    nppiErodeBorder_16u_C3R_Ctx
);

impl_morphology_mask_border!(
    "Four-channel 16-bit unsigned integer erosion with border control.",
    erode_border_u16_c4,
    u16,
    C4,
    nppiErodeBorder_16u_C4R_Ctx
);

impl_morphology_mask_border!(
    "Four-channel 16-bit unsigned integer erosion with border control, ignoring alpha-channel.",
    erode_border_u16_ac4,
    u16,
    AC4,
    nppiErodeBorder_16u_AC4R_Ctx
);

impl_morphology_mask_border!(
    "Single-channel 32-bit floating-point erosion with border control.",
    erode_border_f32_c1,
    f32,
    C1,
    nppiErodeBorder_32f_C1R_Ctx
);

impl_morphology_mask_border!(
    "Three-channel 32-bit floating-point erosion with border control.",
    erode_border_f32_c3,
    f32,
    C3,
    nppiErodeBorder_32f_C3R_Ctx
);

impl_morphology_mask_border!(
    "Four-channel 32-bit floating-point erosion with border control.",
    erode_border_f32_c4,
    f32,
    C4,
    nppiErodeBorder_32f_C4R_Ctx
);

impl_morphology_mask_border!(
    "Four-channel 32-bit floating-point erosion with border control, ignoring alpha-channel.",
    erode_border_f32_ac4,
    f32,
    AC4,
    nppiErodeBorder_32f_AC4R_Ctx
);

impl_generic_morphology_mask_border!(
    "Performs dilation using `mask`, `mask_size`, and `anchor` with border control.",
DilateBorderC1, dilate_border, dilate_border_c1, C1, [
    u8 => dilate_border_u8_c1,
    u16 => dilate_border_u16_c1,
    f32 => dilate_border_f32_c1,
]
);

impl_generic_morphology_mask_border!(
    "Performs dilation using `mask`, `mask_size`, and `anchor` with border control.",
DilateBorderC3, dilate_border, dilate_border_c3, C3, [
    u8 => dilate_border_u8_c3,
    u16 => dilate_border_u16_c3,
    f32 => dilate_border_f32_c3,
]
);

impl_generic_morphology_mask_border!(
    "Performs dilation using `mask`, `mask_size`, and `anchor` with border control.",
DilateBorderC4, dilate_border, dilate_border_c4, C4, [
    u8 => dilate_border_u8_c4,
    u16 => dilate_border_u16_c4,
    f32 => dilate_border_f32_c4,
]
);

impl_generic_morphology_mask_border!(
    "Performs dilation using `mask`, `mask_size`, and `anchor` with border control. The destination alpha channel is not processed.",
DilateBorderAc4, dilate_border, dilate_border_ac4, AC4, [
    u8 => dilate_border_u8_ac4,
    u16 => dilate_border_u16_ac4,
    f32 => dilate_border_f32_ac4,
]
);

impl_generic_morphology_mask_border!(
    "Performs erosion using `mask`, `mask_size`, and `anchor` with border control.",
ErodeBorderC1, erode_border, erode_border_c1, C1, [
    u8 => erode_border_u8_c1,
    u16 => erode_border_u16_c1,
    f32 => erode_border_f32_c1,
]
);

impl_generic_morphology_mask_border!(
    "Performs erosion using `mask`, `mask_size`, and `anchor` with border control.",
ErodeBorderC3, erode_border, erode_border_c3, C3, [
    u8 => erode_border_u8_c3,
    u16 => erode_border_u16_c3,
    f32 => erode_border_f32_c3,
]
);

impl_generic_morphology_mask_border!(
    "Performs erosion using `mask`, `mask_size`, and `anchor` with border control.",
ErodeBorderC4, erode_border, erode_border_c4, C4, [
    u8 => erode_border_u8_c4,
    u16 => erode_border_u16_c4,
    f32 => erode_border_f32_c4,
]
);

impl_generic_morphology_mask_border!(
    "Performs erosion using `mask`, `mask_size`, and `anchor` with border control. The destination alpha channel is not processed.",
ErodeBorderAc4, erode_border, erode_border_ac4, AC4, [
    u8 => erode_border_u8_ac4,
    u16 => erode_border_u16_ac4,
    f32 => erode_border_f32_ac4,
]
);

macro_rules! impl_gray_morphology_mask_border {
    ($doc:literal, $name:ident, $ty:ty, $mask_ty:ty, $ffi:ident) => {
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
        /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Mask storage must satisfy the native structuring-element and address-space requirements and remain readable for as long as native work uses it.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, C1>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $ty, C1>,
            mask: &[$mask_ty],
            mask_size: Size,
            anchor: Point,
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
            validate_mask_shape(mask, mask_size, anchor)?;

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
                    mask.as_ptr().cast(),
                    mask_size.into(),
                    anchor.into(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_generic_gray_morphology_mask_border {
    ($doc:literal, $trait:ident, $method:ident, $function:ident, [$($ty:ty, $mask_ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native image and structuring-element ABI and use a compatible morphology routine.
        /// They must uphold the method's geometry checks, access rules, and queued lifetimes.
        pub unsafe trait $trait: DataTypeLike + Sized {
            /// Native element type used by the grayscale structuring element.
            type Mask: Copy;

            #[doc = $doc]
            ///
            /// Enqueues work on the selected stream.
            /// Row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
            /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source pixels must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Mask storage must satisfy the native structuring-element and address-space requirements and remain readable for as long as native work uses it.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C1>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, C1>,
                mask: &[Self::Mask],
                mask_size: Size,
                anchor: Point,
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
        /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source pixels must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Mask storage must satisfy the native structuring-element and address-space requirements and remain readable for as long as native work uses it.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C1>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, T, C1>,
            mask: &[T::Mask],
            mask_size: Size,
            anchor: Point,
            border_type: BorderType,
        ) -> Result<()>
        where
            T: $trait,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask,
                    mask_size,
                    anchor,
                    border_type,
                )
            }
        }

        $(
            unsafe impl $trait for $ty {
                type Mask = $mask_ty;

                unsafe fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C1>,
                    source_offset: Point,
                    destination: &mut ImageOutputView<'_, Self, C1>,
                    mask: &[Self::Mask],
                    mask_size: Size,
                    anchor: Point,
                    border_type: BorderType,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct(
                            stream_context,
                            source,
                            source_offset,
                            destination,
                            mask,
                            mask_size,
                            anchor,
                            border_type,
                        )
                    }
                }
            }
        )*
    };
}

impl_gray_morphology_mask_border!(
    "Single-channel 8-bit unsigned integer gray scale dilation with border control.",
    gray_dilate_border_u8_c1,
    u8,
    i32,
    nppiGrayDilateBorder_8u_C1R_Ctx
);

impl_gray_morphology_mask_border!(
    "Single-channel 32-bit floating point gray scale dilation with border control.",
    gray_dilate_border_f32_c1,
    f32,
    f32,
    nppiGrayDilateBorder_32f_C1R_Ctx
);

impl_gray_morphology_mask_border!(
    "Single-channel 8-bit unsigned integer gray scale erosion with border control.",
    gray_erode_border_u8_c1,
    u8,
    i32,
    nppiGrayErodeBorder_8u_C1R_Ctx
);

impl_gray_morphology_mask_border!(
    "Single-channel 32-bit floating point gray scale erosion with border control.",
    gray_erode_border_f32_c1,
    f32,
    f32,
    nppiGrayErodeBorder_32f_C1R_Ctx
);

impl_generic_gray_morphology_mask_border!(
    "Performs grayscale dilation using `mask`, `mask_size`, and `anchor` with border control.",
GrayDilateBorderC1, gray_dilate_border, gray_dilate_border_c1, [
    u8, i32 => gray_dilate_border_u8_c1,
    f32, f32 => gray_dilate_border_f32_c1,
]
);

impl_generic_gray_morphology_mask_border!(
    "Performs grayscale erosion using `mask`, `mask_size`, and `anchor` with border control.",
GrayErodeBorderC1, gray_erode_border, gray_erode_border_c1, [
    u8, i32 => gray_erode_border_u8_c1,
    f32, f32 => gray_erode_border_f32_c1,
]
);
