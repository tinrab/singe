use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        morphology::{validate_border_roi, validate_mask, validate_positive_size},
        view::{C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point, Size},
    utility::to_usize,
    workspace::ScratchBuffer,
};

macro_rules! impl_morph_buffer_size {
    ($name:ident, $ffi:ident) => {
        /// Returns the scratch size in bytes for composite morphology on `roi`.
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if either ROI dimension is nonpositive, NPP rejects the query, or its returned size cannot be represented as `usize`.
        pub fn $name(roi: Size) -> Result<usize> {
            validate_positive_size(roi, "roi")?;

            let mut bytes = 0;
            // SAFETY: the `ffi` entry point receives valid output locations.
            unsafe {
                try_ffi!(sys::$ffi(roi.into(), &raw mut bytes))?;
            }

            to_usize(bytes, "buffer size")
        }
    };
}

macro_rules! impl_generic_morph_buffer_size {
    ($trait:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and uphold the methods' validation and execution contracts.
        /// Workspace-size queries must not underestimate the scratch required by the matching native operation.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            /// Returns the scratch size in bytes for composite morphology on `roi`.
            ///
            /// # Errors
            ///
            /// Returns an error if either ROI dimension is nonpositive, NPP rejects the query, or its returned size cannot be represented as `usize`.
            fn morph_buffer_size(roi: Size) -> Result<usize>;
        }

        /// Returns the scratch size in bytes for composite morphology on `roi`.
        ///
        /// # Errors
        ///
        /// Returns an error if either ROI dimension is nonpositive, NPP rejects the query, or its returned size cannot be represented as `usize`.
        pub fn $function<T>(roi: Size) -> Result<usize>
        where
            T: $trait<$layout>,
        {
            T::morph_buffer_size(roi)
        }

        $(
            unsafe impl $trait<$layout> for $ty {
                fn morph_buffer_size(roi: Size) -> Result<usize> {
                    $direct(roi)
                }
            }
        )*
    };
}

impl_morph_buffer_size!(morph_buffer_size_u8_c1, nppiMorphGetBufferSize_8u_C1R);

impl_morph_buffer_size!(morph_buffer_size_u8_c3, nppiMorphGetBufferSize_8u_C3R);

impl_morph_buffer_size!(morph_buffer_size_u8_c4, nppiMorphGetBufferSize_8u_C4R);

impl_morph_buffer_size!(morph_buffer_size_u16_c1, nppiMorphGetBufferSize_16u_C1R);

impl_morph_buffer_size!(morph_buffer_size_i16_c1, nppiMorphGetBufferSize_16s_C1R);

impl_morph_buffer_size!(morph_buffer_size_f32_c1, nppiMorphGetBufferSize_32f_C1R);

impl_morph_buffer_size!(morph_buffer_size_f32_c3, nppiMorphGetBufferSize_32f_C3R);

impl_morph_buffer_size!(morph_buffer_size_f32_c4, nppiMorphGetBufferSize_32f_C4R);

impl_generic_morph_buffer_size!(MorphBufferSizeC1, morph_buffer_size_c1, C1, [
    u8 => morph_buffer_size_u8_c1,
    u16 => morph_buffer_size_u16_c1,
    i16 => morph_buffer_size_i16_c1,
    f32 => morph_buffer_size_f32_c1,
]);

impl_generic_morph_buffer_size!(MorphBufferSizeC3, morph_buffer_size_c3, C3, [
    u8 => morph_buffer_size_u8_c3,
    f32 => morph_buffer_size_f32_c3,
]);

impl_generic_morph_buffer_size!(MorphBufferSizeC4, morph_buffer_size_c4, C4, [
    u8 => morph_buffer_size_u8_c4,
    f32 => morph_buffer_size_f32_c4,
]);

macro_rules! impl_morphology_composite_border {
    ($doc:literal, $with_scratch_name:ident, $name:ident, $ty:ty, $layout:ty, $buffer_size:ident, $ffi:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Submits work on the selected stream.
        /// Uses caller-provided scratch.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
        /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
        /// The scratch-size query can also fail.
        /// Returns an error if scratch is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Mask storage must satisfy the native address-space requirements and remain readable for as long as native work uses it.
        /// Scratch must belong to the selected CUDA context, remain exclusively writable and disjoint from images and the mask, and outlive queued native use.
        /// This wrapper checks scratch capacity but does not check its CUDA context.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $with_scratch_name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $ty, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            mask: &[u8],
            mask_size: Size,
            anchor: Point,
            scratch: &mut ScratchBuffer,
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
            scratch.require($buffer_size(destination.size())?)?;

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
                    scratch.as_mut_ptr().cast(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }

        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// Submits work on the selected stream.
        /// Allocates temporary scratch using the corresponding ROI-size query.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
        /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
        /// The scratch-size query can also fail.
        /// Scratch allocation can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Mask storage must satisfy the native address-space requirements and remain readable for as long as native work uses it.
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
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                let mut scratch = ScratchBuffer::create(
                    stream_context.cuda_context(),
                    $buffer_size(destination.size())?,
                )?;
                $with_scratch_name(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask,
                    mask_size,
                    anchor,
                    &mut scratch,
                    border_type,
                )
            }
        }
    };
}

macro_rules! impl_generic_morphology_composite_border {
    (
        $doc:literal,
        $trait:ident,
        $method_with_scratch:ident,
        $method:ident,
        $function_with_scratch:ident,
        $function:ident,
        $layout:ty,
        [$($ty:ty => $direct_with_scratch:ident, $direct:ident),* $(,)?]
    ) => {
        #[doc = concat!("Native element and layout dispatch for [`", stringify!($function), "`].")]
        ///
        /// # Safety
        ///
        /// Implementations must preserve the native element and channel-layout ABI and uphold the methods' validation and execution contracts.
        /// Workspace-size queries must not underestimate the scratch required by the matching native operation.
        pub unsafe trait $trait<Layout>: DataTypeLike + Sized {
            #[doc = $doc]
            ///
            /// Submits work on the selected stream.
            /// Uses caller-provided scratch.
            /// Row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
            /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
            /// The scratch-size query can also fail.
            /// Returns an error if scratch is smaller than the queried requirement.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Mask storage must satisfy the native address-space requirements and remain readable for as long as native work uses it.
            /// Scratch must belong to the selected CUDA context, remain exclusively writable and disjoint from images and the mask, and outlive queued native use.
            /// This wrapper checks scratch capacity but does not check its CUDA context.
            /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
            unsafe fn $method_with_scratch(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, Layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, Layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
                scratch: &mut ScratchBuffer,
                border_type: BorderType,
            ) -> Result<()>;

            #[doc = $doc]
            ///
            /// Submits work on the selected stream.
            /// Allocates temporary scratch using the corresponding ROI-size query.
            /// Row padding remains unwritten.
            ///
            /// # Errors
            ///
            /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
            /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
            /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
            /// The scratch-size query can also fail.
            /// Scratch allocation can also fail.
            ///
            /// # Safety
            ///
            /// Follow the [raw image execution contract](crate::image::raw).
            /// Source samples must be initialized and ready when read by the selected stream.
            /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
            /// Retain image allocations through queued completion, including submission errors and captured graph replay.
            /// Mask storage must satisfy the native address-space requirements and remain readable for as long as native work uses it.
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
        /// Submits work on the selected stream.
        /// Uses caller-provided scratch.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
        /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
        /// The scratch-size query can also fail.
        /// Returns an error if scratch is smaller than the queried requirement.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Mask storage must satisfy the native address-space requirements and remain readable for as long as native work uses it.
        /// Scratch must belong to the selected CUDA context, remain exclusively writable and disjoint from images and the mask, and outlive queued native use.
        /// This wrapper checks scratch capacity but does not check its CUDA context.
        /// Image layouts, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $function_with_scratch<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $layout>,
            source_offset: Point,
            destination: &mut ImageOutputView<'_, T, $layout>,
            mask: &[u8],
            mask_size: Size,
            anchor: Point,
            scratch: &mut ScratchBuffer,
            border_type: BorderType,
        ) -> Result<()>
        where
            T: $trait<$layout>,
        {
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                T::$method_with_scratch(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask,
                    mask_size,
                    anchor,
                    scratch,
                    border_type,
                )
            }
        }

        #[doc = $doc]
        ///
        /// Submits work on the selected stream.
        /// Allocates temporary scratch using the corresponding ROI-size query.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the operation parameters.
        /// `source_offset` must address a source pixel, ROI endpoints must not overflow, and the destination-sized ROI must fit within the source image.
        /// Mask dimensions must be positive, their checked product must equal `mask.len()`, and `anchor` must lie within the mask.
        /// The scratch-size query can also fail.
        /// Scratch allocation can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Source samples must be initialized and ready when read by the selected stream.
        /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
        /// Retain image allocations through queued completion, including submission errors and captured graph replay.
        /// Mask storage must satisfy the native address-space requirements and remain readable for as long as native work uses it.
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
                unsafe fn $method_with_scratch(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $layout>,
                    source_offset: Point,
                    destination: &mut ImageOutputView<'_, Self, $layout>,
                    mask: &[u8],
                    mask_size: Size,
                    anchor: Point,
                    scratch: &mut ScratchBuffer,
                    border_type: BorderType,
                ) -> Result<()> {
                    // SAFETY: the caller upholds this function's safety contract.
                    unsafe {
                        $direct_with_scratch(
                            stream_context,
                            source,
                            source_offset,
                            destination,
                            mask,
                            mask_size,
                            anchor,
                            scratch,
                            border_type,
                        )
                    }
                }

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

impl_morphology_composite_border!(
    "Performs morphological closing with border control.",
    morph_close_border_u8_c1_with_scratch,
    morph_close_border_u8_c1,
    u8,
    C1,
    morph_buffer_size_u8_c1,
    nppiMorphCloseBorder_8u_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological closing with border control.",
    morph_close_border_u8_c3_with_scratch,
    morph_close_border_u8_c3,
    u8,
    C3,
    morph_buffer_size_u8_c3,
    nppiMorphCloseBorder_8u_C3R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological closing with border control.",
    morph_close_border_u8_c4_with_scratch,
    morph_close_border_u8_c4,
    u8,
    C4,
    morph_buffer_size_u8_c4,
    nppiMorphCloseBorder_8u_C4R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological closing with border control.",
    morph_close_border_u16_c1_with_scratch,
    morph_close_border_u16_c1,
    u16,
    C1,
    morph_buffer_size_u16_c1,
    nppiMorphCloseBorder_16u_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological closing with border control.",
    morph_close_border_i16_c1_with_scratch,
    morph_close_border_i16_c1,
    i16,
    C1,
    morph_buffer_size_i16_c1,
    nppiMorphCloseBorder_16s_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological closing with border control.",
    morph_close_border_f32_c1_with_scratch,
    morph_close_border_f32_c1,
    f32,
    C1,
    morph_buffer_size_f32_c1,
    nppiMorphCloseBorder_32f_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological closing with border control.",
    morph_close_border_f32_c3_with_scratch,
    morph_close_border_f32_c3,
    f32,
    C3,
    morph_buffer_size_f32_c3,
    nppiMorphCloseBorder_32f_C3R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological closing with border control.",
    morph_close_border_f32_c4_with_scratch,
    morph_close_border_f32_c4,
    f32,
    C4,
    morph_buffer_size_f32_c4,
    nppiMorphCloseBorder_32f_C4R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological opening with border control.",
    morph_open_border_u8_c1_with_scratch,
    morph_open_border_u8_c1,
    u8,
    C1,
    morph_buffer_size_u8_c1,
    nppiMorphOpenBorder_8u_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological opening with border control.",
    morph_open_border_u8_c3_with_scratch,
    morph_open_border_u8_c3,
    u8,
    C3,
    morph_buffer_size_u8_c3,
    nppiMorphOpenBorder_8u_C3R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological opening with border control.",
    morph_open_border_u8_c4_with_scratch,
    morph_open_border_u8_c4,
    u8,
    C4,
    morph_buffer_size_u8_c4,
    nppiMorphOpenBorder_8u_C4R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological opening with border control.",
    morph_open_border_u16_c1_with_scratch,
    morph_open_border_u16_c1,
    u16,
    C1,
    morph_buffer_size_u16_c1,
    nppiMorphOpenBorder_16u_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological opening with border control.",
    morph_open_border_i16_c1_with_scratch,
    morph_open_border_i16_c1,
    i16,
    C1,
    morph_buffer_size_i16_c1,
    nppiMorphOpenBorder_16s_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological opening with border control.",
    morph_open_border_f32_c1_with_scratch,
    morph_open_border_f32_c1,
    f32,
    C1,
    morph_buffer_size_f32_c1,
    nppiMorphOpenBorder_32f_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological opening with border control.",
    morph_open_border_f32_c3_with_scratch,
    morph_open_border_f32_c3,
    f32,
    C3,
    morph_buffer_size_f32_c3,
    nppiMorphOpenBorder_32f_C3R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological opening with border control.",
    morph_open_border_f32_c4_with_scratch,
    morph_open_border_f32_c4,
    f32,
    C4,
    morph_buffer_size_f32_c4,
    nppiMorphOpenBorder_32f_C4R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological top hat with border control.",
    morph_top_hat_border_u8_c1_with_scratch,
    morph_top_hat_border_u8_c1,
    u8,
    C1,
    morph_buffer_size_u8_c1,
    nppiMorphTopHatBorder_8u_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological top hat with border control.",
    morph_top_hat_border_u8_c3_with_scratch,
    morph_top_hat_border_u8_c3,
    u8,
    C3,
    morph_buffer_size_u8_c3,
    nppiMorphTopHatBorder_8u_C3R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological top hat with border control.",
    morph_top_hat_border_u8_c4_with_scratch,
    morph_top_hat_border_u8_c4,
    u8,
    C4,
    morph_buffer_size_u8_c4,
    nppiMorphTopHatBorder_8u_C4R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological top hat with border control.",
    morph_top_hat_border_u16_c1_with_scratch,
    morph_top_hat_border_u16_c1,
    u16,
    C1,
    morph_buffer_size_u16_c1,
    nppiMorphTopHatBorder_16u_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological top hat with border control.",
    morph_top_hat_border_i16_c1_with_scratch,
    morph_top_hat_border_i16_c1,
    i16,
    C1,
    morph_buffer_size_i16_c1,
    nppiMorphTopHatBorder_16s_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological top hat with border control.",
    morph_top_hat_border_f32_c1_with_scratch,
    morph_top_hat_border_f32_c1,
    f32,
    C1,
    morph_buffer_size_f32_c1,
    nppiMorphTopHatBorder_32f_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological top hat with border control.",
    morph_top_hat_border_f32_c3_with_scratch,
    morph_top_hat_border_f32_c3,
    f32,
    C3,
    morph_buffer_size_f32_c3,
    nppiMorphTopHatBorder_32f_C3R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological top hat with border control.",
    morph_top_hat_border_f32_c4_with_scratch,
    morph_top_hat_border_f32_c4,
    f32,
    C4,
    morph_buffer_size_f32_c4,
    nppiMorphTopHatBorder_32f_C4R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological black hat with border control.",
    morph_black_hat_border_u8_c1_with_scratch,
    morph_black_hat_border_u8_c1,
    u8,
    C1,
    morph_buffer_size_u8_c1,
    nppiMorphBlackHatBorder_8u_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological black hat with border control.",
    morph_black_hat_border_u8_c3_with_scratch,
    morph_black_hat_border_u8_c3,
    u8,
    C3,
    morph_buffer_size_u8_c3,
    nppiMorphBlackHatBorder_8u_C3R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological black hat with border control.",
    morph_black_hat_border_u8_c4_with_scratch,
    morph_black_hat_border_u8_c4,
    u8,
    C4,
    morph_buffer_size_u8_c4,
    nppiMorphBlackHatBorder_8u_C4R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological black hat with border control.",
    morph_black_hat_border_u16_c1_with_scratch,
    morph_black_hat_border_u16_c1,
    u16,
    C1,
    morph_buffer_size_u16_c1,
    nppiMorphBlackHatBorder_16u_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological black hat with border control.",
    morph_black_hat_border_i16_c1_with_scratch,
    morph_black_hat_border_i16_c1,
    i16,
    C1,
    morph_buffer_size_i16_c1,
    nppiMorphBlackHatBorder_16s_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological black hat with border control.",
    morph_black_hat_border_f32_c1_with_scratch,
    morph_black_hat_border_f32_c1,
    f32,
    C1,
    morph_buffer_size_f32_c1,
    nppiMorphBlackHatBorder_32f_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological black hat with border control.",
    morph_black_hat_border_f32_c3_with_scratch,
    morph_black_hat_border_f32_c3,
    f32,
    C3,
    morph_buffer_size_f32_c3,
    nppiMorphBlackHatBorder_32f_C3R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological black hat with border control.",
    morph_black_hat_border_f32_c4_with_scratch,
    morph_black_hat_border_f32_c4,
    f32,
    C4,
    morph_buffer_size_f32_c4,
    nppiMorphBlackHatBorder_32f_C4R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological gradient with border control.",
    morph_gradient_border_u8_c1_with_scratch,
    morph_gradient_border_u8_c1,
    u8,
    C1,
    morph_buffer_size_u8_c1,
    nppiMorphGradientBorder_8u_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological gradient with border control.",
    morph_gradient_border_u8_c3_with_scratch,
    morph_gradient_border_u8_c3,
    u8,
    C3,
    morph_buffer_size_u8_c3,
    nppiMorphGradientBorder_8u_C3R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological gradient with border control.",
    morph_gradient_border_u8_c4_with_scratch,
    morph_gradient_border_u8_c4,
    u8,
    C4,
    morph_buffer_size_u8_c4,
    nppiMorphGradientBorder_8u_C4R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological gradient with border control.",
    morph_gradient_border_u16_c1_with_scratch,
    morph_gradient_border_u16_c1,
    u16,
    C1,
    morph_buffer_size_u16_c1,
    nppiMorphGradientBorder_16u_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological gradient with border control.",
    morph_gradient_border_i16_c1_with_scratch,
    morph_gradient_border_i16_c1,
    i16,
    C1,
    morph_buffer_size_i16_c1,
    nppiMorphGradientBorder_16s_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological gradient with border control.",
    morph_gradient_border_f32_c1_with_scratch,
    morph_gradient_border_f32_c1,
    f32,
    C1,
    morph_buffer_size_f32_c1,
    nppiMorphGradientBorder_32f_C1R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological gradient with border control.",
    morph_gradient_border_f32_c3_with_scratch,
    morph_gradient_border_f32_c3,
    f32,
    C3,
    morph_buffer_size_f32_c3,
    nppiMorphGradientBorder_32f_C3R_Ctx
);

impl_morphology_composite_border!(
    "Performs morphological gradient with border control.",
    morph_gradient_border_f32_c4_with_scratch,
    morph_gradient_border_f32_c4,
    f32,
    C4,
    morph_buffer_size_f32_c4,
    nppiMorphGradientBorder_32f_C4R_Ctx
);

impl_generic_morphology_composite_border!(
    "Performs morphological closing with border control.",
MorphCloseBorderC1,
    morph_close_border_with_scratch,
    morph_close_border,
    morph_close_border_c1_with_scratch,
    morph_close_border_c1,
    C1,
    [
        u8 => morph_close_border_u8_c1_with_scratch, morph_close_border_u8_c1,
        u16 => morph_close_border_u16_c1_with_scratch, morph_close_border_u16_c1,
        i16 => morph_close_border_i16_c1_with_scratch, morph_close_border_i16_c1,
        f32 => morph_close_border_f32_c1_with_scratch, morph_close_border_f32_c1,
    ]
);

impl_generic_morphology_composite_border!(
    "Performs morphological closing with border control.",
MorphCloseBorderC3,
    morph_close_border_with_scratch,
    morph_close_border,
    morph_close_border_c3_with_scratch,
    morph_close_border_c3,
    C3,
    [
        u8 => morph_close_border_u8_c3_with_scratch, morph_close_border_u8_c3,
        f32 => morph_close_border_f32_c3_with_scratch, morph_close_border_f32_c3,
    ]
);

impl_generic_morphology_composite_border!(
    "Performs morphological closing with border control.",
MorphCloseBorderC4,
    morph_close_border_with_scratch,
    morph_close_border,
    morph_close_border_c4_with_scratch,
    morph_close_border_c4,
    C4,
    [
        u8 => morph_close_border_u8_c4_with_scratch, morph_close_border_u8_c4,
        f32 => morph_close_border_f32_c4_with_scratch, morph_close_border_f32_c4,
    ]
);

impl_generic_morphology_composite_border!(
    "Performs morphological opening with border control.",
MorphOpenBorderC1,
    morph_open_border_with_scratch,
    morph_open_border,
    morph_open_border_c1_with_scratch,
    morph_open_border_c1,
    C1,
    [
        u8 => morph_open_border_u8_c1_with_scratch, morph_open_border_u8_c1,
        u16 => morph_open_border_u16_c1_with_scratch, morph_open_border_u16_c1,
        i16 => morph_open_border_i16_c1_with_scratch, morph_open_border_i16_c1,
        f32 => morph_open_border_f32_c1_with_scratch, morph_open_border_f32_c1,
    ]
);

impl_generic_morphology_composite_border!(
    "Performs morphological opening with border control.",
MorphOpenBorderC3,
    morph_open_border_with_scratch,
    morph_open_border,
    morph_open_border_c3_with_scratch,
    morph_open_border_c3,
    C3,
    [
        u8 => morph_open_border_u8_c3_with_scratch, morph_open_border_u8_c3,
        f32 => morph_open_border_f32_c3_with_scratch, morph_open_border_f32_c3,
    ]
);

impl_generic_morphology_composite_border!(
    "Performs morphological opening with border control.",
MorphOpenBorderC4,
    morph_open_border_with_scratch,
    morph_open_border,
    morph_open_border_c4_with_scratch,
    morph_open_border_c4,
    C4,
    [
        u8 => morph_open_border_u8_c4_with_scratch, morph_open_border_u8_c4,
        f32 => morph_open_border_f32_c4_with_scratch, morph_open_border_f32_c4,
    ]
);

impl_generic_morphology_composite_border!(
    "Performs morphological top hat with border control.",
MorphTopHatBorderC1,
    morph_top_hat_border_with_scratch,
    morph_top_hat_border,
    morph_top_hat_border_c1_with_scratch,
    morph_top_hat_border_c1,
    C1,
    [
        u8 => morph_top_hat_border_u8_c1_with_scratch, morph_top_hat_border_u8_c1,
        u16 => morph_top_hat_border_u16_c1_with_scratch, morph_top_hat_border_u16_c1,
        i16 => morph_top_hat_border_i16_c1_with_scratch, morph_top_hat_border_i16_c1,
        f32 => morph_top_hat_border_f32_c1_with_scratch, morph_top_hat_border_f32_c1,
    ]
);

impl_generic_morphology_composite_border!(
    "Performs morphological top hat with border control.",
MorphTopHatBorderC3,
    morph_top_hat_border_with_scratch,
    morph_top_hat_border,
    morph_top_hat_border_c3_with_scratch,
    morph_top_hat_border_c3,
    C3,
    [
        u8 => morph_top_hat_border_u8_c3_with_scratch, morph_top_hat_border_u8_c3,
        f32 => morph_top_hat_border_f32_c3_with_scratch, morph_top_hat_border_f32_c3,
    ]
);

impl_generic_morphology_composite_border!(
    "Performs morphological top hat with border control.",
MorphTopHatBorderC4,
    morph_top_hat_border_with_scratch,
    morph_top_hat_border,
    morph_top_hat_border_c4_with_scratch,
    morph_top_hat_border_c4,
    C4,
    [
        u8 => morph_top_hat_border_u8_c4_with_scratch, morph_top_hat_border_u8_c4,
        f32 => morph_top_hat_border_f32_c4_with_scratch, morph_top_hat_border_f32_c4,
    ]
);

impl_generic_morphology_composite_border!(
    "Performs morphological black hat with border control.",
MorphBlackHatBorderC1,
    morph_black_hat_border_with_scratch,
    morph_black_hat_border,
    morph_black_hat_border_c1_with_scratch,
    morph_black_hat_border_c1,
    C1,
    [
        u8 => morph_black_hat_border_u8_c1_with_scratch, morph_black_hat_border_u8_c1,
        u16 => morph_black_hat_border_u16_c1_with_scratch, morph_black_hat_border_u16_c1,
        i16 => morph_black_hat_border_i16_c1_with_scratch, morph_black_hat_border_i16_c1,
        f32 => morph_black_hat_border_f32_c1_with_scratch, morph_black_hat_border_f32_c1,
    ]
);

impl_generic_morphology_composite_border!(
    "Performs morphological black hat with border control.",
MorphBlackHatBorderC3,
    morph_black_hat_border_with_scratch,
    morph_black_hat_border,
    morph_black_hat_border_c3_with_scratch,
    morph_black_hat_border_c3,
    C3,
    [
        u8 => morph_black_hat_border_u8_c3_with_scratch, morph_black_hat_border_u8_c3,
        f32 => morph_black_hat_border_f32_c3_with_scratch, morph_black_hat_border_f32_c3,
    ]
);

impl_generic_morphology_composite_border!(
    "Performs morphological black hat with border control.",
MorphBlackHatBorderC4,
    morph_black_hat_border_with_scratch,
    morph_black_hat_border,
    morph_black_hat_border_c4_with_scratch,
    morph_black_hat_border_c4,
    C4,
    [
        u8 => morph_black_hat_border_u8_c4_with_scratch, morph_black_hat_border_u8_c4,
        f32 => morph_black_hat_border_f32_c4_with_scratch, morph_black_hat_border_f32_c4,
    ]
);

impl_generic_morphology_composite_border!(
    "Performs morphological gradient with border control.",
MorphGradientBorderC1,
    morph_gradient_border_with_scratch,
    morph_gradient_border,
    morph_gradient_border_c1_with_scratch,
    morph_gradient_border_c1,
    C1,
    [
        u8 => morph_gradient_border_u8_c1_with_scratch, morph_gradient_border_u8_c1,
        u16 => morph_gradient_border_u16_c1_with_scratch, morph_gradient_border_u16_c1,
        i16 => morph_gradient_border_i16_c1_with_scratch, morph_gradient_border_i16_c1,
        f32 => morph_gradient_border_f32_c1_with_scratch, morph_gradient_border_f32_c1,
    ]
);

impl_generic_morphology_composite_border!(
    "Performs morphological gradient with border control.",
MorphGradientBorderC3,
    morph_gradient_border_with_scratch,
    morph_gradient_border,
    morph_gradient_border_c3_with_scratch,
    morph_gradient_border_c3,
    C3,
    [
        u8 => morph_gradient_border_u8_c3_with_scratch, morph_gradient_border_u8_c3,
        f32 => morph_gradient_border_f32_c3_with_scratch, morph_gradient_border_f32_c3,
    ]
);

impl_generic_morphology_composite_border!(
    "Performs morphological gradient with border control.",
MorphGradientBorderC4,
    morph_gradient_border_with_scratch,
    morph_gradient_border,
    morph_gradient_border_c4_with_scratch,
    morph_gradient_border_c4,
    C4,
    [
        u8 => morph_gradient_border_u8_c4_with_scratch, morph_gradient_border_u8_c4,
        f32 => morph_gradient_border_f32_c4_with_scratch, morph_gradient_border_f32_c4,
    ]
);
