use singe_cuda::types::f16;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic_validation::validate_same_size,
        view::{C1, ImageView, ImageViewMut, MaskView},
    },
    try_ffi,
};

macro_rules! impl_accumulate_square {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        /// Submits a single-channel sum-of-squares update to the selected stream.
        ///
        /// Adds the square of each `source` pixel to the corresponding pixel in
        /// `source_destination`. Both images must have the same size.
        ///
        /// The call does not wait for GPU completion.
        ///
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if the CUDA context cannot be bound, an image belongs to
        /// another context, image sizes differ, or NPP returns a non-success status.
        ///
        /// # Safety
        ///
        /// Order earlier writes to each source image before this stream
        /// reads them. Keep their allocations readable and unchanged until the
        /// operation completes. The destination must have exclusive read/write
        /// access during the operation and must not overlap a source.
        /// Retain the stream context and all image allocations until completion,
        /// including after errors. Captured operations require the allocations and
        /// access guarantees to remain valid through every graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            source_destination: &mut ImageViewMut<'_, $destination_ty, C1>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(source_destination)?;

            validate_same_size(source.size(), source_destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_accumulate_square_masked {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        /// Submits a masked single-channel sum-of-squares update.
        ///
        /// Adds the square of each `source` pixel to `source_destination` only where
        /// the corresponding `mask` byte is nonzero. Other destination pixels keep
        /// their existing values. All three images must have the same size.
        ///
        /// The call does not wait for GPU completion.
        ///
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if the CUDA context cannot be bound, an image belongs to
        /// another context, image sizes differ, or NPP returns a non-success status.
        ///
        /// # Safety
        ///
        /// Order earlier writes to each source image and mask before this stream
        /// reads them. Keep their allocations readable and unchanged until the
        /// operation completes. The destination must have exclusive read/write
        /// access during the operation and must not overlap a source or mask.
        /// Retain the stream context and all image allocations until completion,
        /// including after errors. Captured operations require the allocations and
        /// access guarantees to remain valid through every graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            mask: &MaskView<'_>,
            source_destination: &mut ImageViewMut<'_, $destination_ty, C1>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mask)?;
            stream_context.validate_region(source_destination)?;

            validate_same_size(source.size(), mask.size())?;
            validate_same_size(source.size(), source_destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_accumulate_product {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        /// Submits a single-channel product accumulation to the selected stream.
        ///
        /// Adds the pixelwise product of `source1` and `source2` to
        /// `source_destination`. All three images must have the same size.
        ///
        /// The call does not wait for GPU completion.
        ///
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if the CUDA context cannot be bound, an image belongs to
        /// another context, image sizes differ, or NPP returns a non-success status.
        ///
        /// # Safety
        ///
        /// Order earlier writes to each source image before this stream
        /// reads them. Keep their allocations readable and unchanged until the
        /// operation completes. The destination must have exclusive read/write
        /// access during the operation and must not overlap a source.
        /// Retain the stream context and all image allocations until completion,
        /// including after errors. Captured operations require the allocations and
        /// access guarantees to remain valid through every graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $source_ty, C1>,
            source2: &ImageView<'_, $source_ty, C1>,
            source_destination: &mut ImageViewMut<'_, $destination_ty, C1>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source1)?;
            stream_context.validate_region(source2)?;
            stream_context.validate_region(source_destination)?;

            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), source_destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source1.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_accumulate_product_masked {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        /// Submits a masked single-channel product accumulation.
        ///
        /// Adds the pixelwise product of `source1` and `source2` to
        /// `source_destination` only where `mask` is nonzero. Other destination
        /// pixels keep their existing values. All four images must have the same size.
        ///
        /// The call does not wait for GPU completion.
        ///
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if the CUDA context cannot be bound, an image belongs to
        /// another context, image sizes differ, or NPP returns a non-success status.
        ///
        /// # Safety
        ///
        /// Order earlier writes to each source image and mask before this stream
        /// reads them. Keep their allocations readable and unchanged until the
        /// operation completes. The destination must have exclusive read/write
        /// access during the operation and must not overlap a source or mask.
        /// Retain the stream context and all image allocations until completion,
        /// including after errors. Captured operations require the allocations and
        /// access guarantees to remain valid through every graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source1: &ImageView<'_, $source_ty, C1>,
            source2: &ImageView<'_, $source_ty, C1>,
            mask: &MaskView<'_>,
            source_destination: &mut ImageViewMut<'_, $destination_ty, C1>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source1)?;
            stream_context.validate_region(source2)?;
            stream_context.validate_region(mask)?;
            stream_context.validate_region(source_destination)?;

            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), mask.size())?;
            validate_same_size(source1.size(), source_destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source1.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_accumulate_weighted {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        /// Submits a single-channel alpha-weighted accumulation.
        ///
        /// Applies the native weighted-add operation from `source` to
        /// `source_destination`, with source weight `alpha` in `0.0..=1.0`. Both
        /// images must have the same size.
        ///
        /// The call does not wait for GPU completion.
        ///
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if the CUDA context cannot be bound, an image belongs to
        /// another context, image sizes differ, or NPP returns a non-success status.
        ///
        /// # Safety
        ///
        /// Order earlier writes to each source image before this stream
        /// reads them. Keep their allocations readable and unchanged until the
        /// operation completes. The destination must have exclusive read/write
        /// access during the operation and must not overlap a source.
        /// Retain the stream context and all image allocations until completion,
        /// including after errors. Captured operations require the allocations and
        /// access guarantees to remain valid through every graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            source_destination: &mut ImageViewMut<'_, $destination_ty, C1>,
            alpha: f32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(source_destination)?;

            validate_same_size(source.size(), source_destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_accumulate_weighted_masked {
    ($name:ident, $source_ty:ty, $destination_ty:ty, $ffi:ident) => {
        /// Submits a masked single-channel alpha-weighted accumulation.
        ///
        /// Applies the native weighted-add operation from `source` to
        /// `source_destination` only where `mask` is nonzero, with source weight
        /// `alpha` in `0.0..=1.0`. Other destination pixels keep their existing
        /// values. All three images must have the same size.
        ///
        /// The call does not wait for GPU completion.
        ///
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if the CUDA context cannot be bound, an image belongs to
        /// another context, image sizes differ, or NPP returns a non-success status.
        ///
        /// # Safety
        ///
        /// Order earlier writes to each source image and mask before this stream
        /// reads them. Keep their allocations readable and unchanged until the
        /// operation completes. The destination must have exclusive read/write
        /// access during the operation and must not overlap a source or mask.
        /// Retain the stream context and all image allocations until completion,
        /// including after errors. Captured operations require the allocations and
        /// access guarantees to remain valid through every graph replay.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $source_ty, C1>,
            mask: &MaskView<'_>,
            source_destination: &mut ImageViewMut<'_, $destination_ty, C1>,
            alpha: f32,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source)?;
            stream_context.validate_region(mask)?;
            stream_context.validate_region(source_destination)?;

            validate_same_size(source.size(), mask.size())?;
            validate_same_size(source.size(), source_destination.size())?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source.as_ptr().cast(),
                    source.step(),
                    mask.as_ptr().cast(),
                    mask.step(),
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source.size().into(),
                    alpha,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_accumulate_square!(
    accumulate_square_u8_to_f32_c1,
    u8,
    f32,
    nppiAddSquare_8u32f_C1IR_Ctx
);

impl_accumulate_square_masked!(
    accumulate_square_u8_to_f32_c1_masked,
    u8,
    f32,
    nppiAddSquare_8u32f_C1IMR_Ctx
);

impl_accumulate_square!(
    accumulate_square_u16_to_f32_c1,
    u16,
    f32,
    nppiAddSquare_16u32f_C1IR_Ctx
);

impl_accumulate_square_masked!(
    accumulate_square_u16_to_f32_c1_masked,
    u16,
    f32,
    nppiAddSquare_16u32f_C1IMR_Ctx
);

impl_accumulate_square!(
    accumulate_square_f32_c1,
    f32,
    f32,
    nppiAddSquare_32f_C1IR_Ctx
);

impl_accumulate_square_masked!(
    accumulate_square_f32_c1_masked,
    f32,
    f32,
    nppiAddSquare_32f_C1IMR_Ctx
);

impl_accumulate_product!(
    accumulate_product_u8_to_f32_c1,
    u8,
    f32,
    nppiAddProduct_8u32f_C1IR_Ctx
);

impl_accumulate_product_masked!(
    accumulate_product_u8_to_f32_c1_masked,
    u8,
    f32,
    nppiAddProduct_8u32f_C1IMR_Ctx
);

impl_accumulate_product!(
    accumulate_product_u16_to_f32_c1,
    u16,
    f32,
    nppiAddProduct_16u32f_C1IR_Ctx
);

impl_accumulate_product_masked!(
    accumulate_product_u16_to_f32_c1_masked,
    u16,
    f32,
    nppiAddProduct_16u32f_C1IMR_Ctx
);

impl_accumulate_product!(
    accumulate_product_f32_c1,
    f32,
    f32,
    nppiAddProduct_32f_C1IR_Ctx
);

impl_accumulate_product_masked!(
    accumulate_product_f32_c1_masked,
    f32,
    f32,
    nppiAddProduct_32f_C1IMR_Ctx
);

impl_accumulate_product!(
    accumulate_product_f16_c1,
    f16,
    f16,
    nppiAddProduct_16f_C1IR_Ctx
);

impl_accumulate_weighted!(
    accumulate_weighted_u8_to_f32_c1,
    u8,
    f32,
    nppiAddWeighted_8u32f_C1IR_Ctx
);

impl_accumulate_weighted_masked!(
    accumulate_weighted_u8_to_f32_c1_masked,
    u8,
    f32,
    nppiAddWeighted_8u32f_C1IMR_Ctx
);

impl_accumulate_weighted!(
    accumulate_weighted_u16_to_f32_c1,
    u16,
    f32,
    nppiAddWeighted_16u32f_C1IR_Ctx
);

impl_accumulate_weighted_masked!(
    accumulate_weighted_u16_to_f32_c1_masked,
    u16,
    f32,
    nppiAddWeighted_16u32f_C1IMR_Ctx
);

impl_accumulate_weighted!(
    accumulate_weighted_f32_c1,
    f32,
    f32,
    nppiAddWeighted_32f_C1IR_Ctx
);

impl_accumulate_weighted_masked!(
    accumulate_weighted_f32_c1_masked,
    f32,
    f32,
    nppiAddWeighted_32f_C1IMR_Ctx
);
