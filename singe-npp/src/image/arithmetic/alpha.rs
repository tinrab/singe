use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic_validation::validate_same_size,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    try_ffi,
    types::{AlphaOperation, DataTypeLike},
};

macro_rules! impl_alpha_comp_constant {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        /// Composites `source1` and `source2` into `destination` using the constant
        /// opacities `alpha1` and `alpha2` and the selected `operation`.
        /// Opacity ranges from zero to the maximum channel value for the pixel type.
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
            source1: &ImageView<'_, $ty, $layout>,
            alpha1: $ty,
            source2: &ImageView<'_, $ty, $layout>,
            alpha2: $ty,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            operation: AlphaOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source1)?;
            stream_context.validate_region(source2)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    alpha1,
                    source2.as_ptr().cast(),
                    source2.step(),
                    alpha2,
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_alpha_comp_constant!(alpha_comp_constant_u8_c1, u8, C1, nppiAlphaCompC_8u_C1R_Ctx);

impl_alpha_comp_constant!(alpha_comp_constant_u8_c3, u8, C3, nppiAlphaCompC_8u_C3R_Ctx);

impl_alpha_comp_constant!(alpha_comp_constant_u8_c4, u8, C4, nppiAlphaCompC_8u_C4R_Ctx);

impl_alpha_comp_constant!(
    alpha_comp_constant_u8_ac4,
    u8,
    AC4,
    nppiAlphaCompC_8u_AC4R_Ctx
);

impl_alpha_comp_constant!(alpha_comp_constant_i8_c1, i8, C1, nppiAlphaCompC_8s_C1R_Ctx);

impl_alpha_comp_constant!(
    alpha_comp_constant_u16_c1,
    u16,
    C1,
    nppiAlphaCompC_16u_C1R_Ctx
);

impl_alpha_comp_constant!(
    alpha_comp_constant_u16_c3,
    u16,
    C3,
    nppiAlphaCompC_16u_C3R_Ctx
);

impl_alpha_comp_constant!(
    alpha_comp_constant_u16_c4,
    u16,
    C4,
    nppiAlphaCompC_16u_C4R_Ctx
);

impl_alpha_comp_constant!(
    alpha_comp_constant_u16_ac4,
    u16,
    AC4,
    nppiAlphaCompC_16u_AC4R_Ctx
);

impl_alpha_comp_constant!(
    alpha_comp_constant_i16_c1,
    i16,
    C1,
    nppiAlphaCompC_16s_C1R_Ctx
);

impl_alpha_comp_constant!(
    alpha_comp_constant_u32_c1,
    u32,
    C1,
    nppiAlphaCompC_32u_C1R_Ctx
);

impl_alpha_comp_constant!(
    alpha_comp_constant_i32_c1,
    i32,
    C1,
    nppiAlphaCompC_32s_C1R_Ctx
);

impl_alpha_comp_constant!(
    alpha_comp_constant_f32_c1,
    f32,
    C1,
    nppiAlphaCompC_32f_C1R_Ctx
);

macro_rules! impl_alpha_comp {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        /// Composites `source1` and `source2` into `destination` using their image alpha
        /// values and the selected `operation`.
        /// Alpha values range from zero to the maximum channel value for the pixel type.
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
            source1: &ImageView<'_, $ty, $layout>,
            source2: &ImageView<'_, $ty, $layout>,
            destination: &mut ImageOutputView<'_, $ty, $layout>,
            operation: AlphaOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source1)?;
            stream_context.validate_region(source2)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_alpha_comp!(alpha_comp_u8_ac1, u8, C1, nppiAlphaComp_8u_AC1R_Ctx);

impl_alpha_comp!(alpha_comp_u8_ac4, u8, AC4, nppiAlphaComp_8u_AC4R_Ctx);

impl_alpha_comp!(alpha_comp_i8_ac1, i8, C1, nppiAlphaComp_8s_AC1R_Ctx);

impl_alpha_comp!(alpha_comp_u16_ac1, u16, C1, nppiAlphaComp_16u_AC1R_Ctx);

impl_alpha_comp!(alpha_comp_u16_ac4, u16, AC4, nppiAlphaComp_16u_AC4R_Ctx);

impl_alpha_comp!(alpha_comp_i16_ac1, i16, C1, nppiAlphaComp_16s_AC1R_Ctx);

impl_alpha_comp!(alpha_comp_u32_ac1, u32, C1, nppiAlphaComp_32u_AC1R_Ctx);

impl_alpha_comp!(alpha_comp_u32_ac4, u32, AC4, nppiAlphaComp_32u_AC4R_Ctx);

impl_alpha_comp!(alpha_comp_i32_ac1, i32, C1, nppiAlphaComp_32s_AC1R_Ctx);

impl_alpha_comp!(alpha_comp_i32_ac4, i32, AC4, nppiAlphaComp_32s_AC4R_Ctx);

impl_alpha_comp!(alpha_comp_f32_ac1, f32, C1, nppiAlphaComp_32f_AC1R_Ctx);

impl_alpha_comp!(alpha_comp_f32_ac4, f32, AC4, nppiAlphaComp_32f_AC4R_Ctx);

macro_rules! impl_comp_color_key_scalar {
    ($name:ident, $ffi:ident) => {
        /// Performs complement color-key replacement of `source1` by `source2`,
        /// using `color_key`, and writes the result to `destination`.
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
            source1: &ImageView<'_, u8, C1>,
            source2: &ImageView<'_, u8, C1>,
            color_key: u8,
            destination: &mut ImageOutputView<'_, u8, C1>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source1)?;
            stream_context.validate_region(source2)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    color_key,
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_comp_color_key_array {
    ($name:ident, $layout:ty, $channels:literal, $ffi:ident) => {
        /// Performs complement color-key replacement of `source1` by `source2`,
        /// using the per-channel `color_key`, and writes the result to `destination`.
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
            source1: &ImageView<'_, u8, $layout>,
            source2: &ImageView<'_, u8, $layout>,
            color_key: [u8; $channels],
            destination: &mut ImageOutputView<'_, u8, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source1)?;
            stream_context.validate_region(source2)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            let mut color_key = color_key;
            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    source2.as_ptr().cast(),
                    source2.step(),
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    color_key.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_alpha_comp_color_key {
    ($name:ident, $ffi:ident) => {
        /// Performs complement color-key replacement of `source1` by `source2` with
        /// alpha blending, writing the result to `destination`.
        /// `alpha1` and `alpha2` are constant opacities in the range 0 through 255.
        /// `operation` selects the composition; premultiplied-alpha operations are excluded.
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
            source1: &ImageView<'_, u8, AC4>,
            alpha1: u8,
            source2: &ImageView<'_, u8, AC4>,
            alpha2: u8,
            color_key: [u8; 4],
            destination: &mut ImageOutputView<'_, u8, AC4>,
            operation: AlphaOperation,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source1)?;
            stream_context.validate_region(source2)?;
            stream_context.validate_region(destination)?;

            validate_same_size(source1.size(), source2.size())?;
            validate_same_size(source1.size(), destination.size())?;

            let mut color_key = color_key;
            // SAFETY: the `ffi` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            unsafe {
                try_ffi!(sys::$ffi(
                    source1.as_ptr().cast(),
                    source1.step(),
                    alpha1,
                    source2.as_ptr().cast(),
                    source2.step(),
                    alpha2,
                    destination.as_mut_ptr().cast(),
                    destination.step(),
                    source1.size().into(),
                    color_key.as_mut_ptr().cast(),
                    operation.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_comp_color_key_scalar!(comp_color_key_u8_c1, nppiCompColorKey_8u_C1R_Ctx);

impl_comp_color_key_array!(comp_color_key_u8_c3, C3, 3, nppiCompColorKey_8u_C3R_Ctx);

impl_comp_color_key_array!(comp_color_key_u8_c4, C4, 4, nppiCompColorKey_8u_C4R_Ctx);

impl_alpha_comp_color_key!(
    alpha_comp_color_key_u8_ac4,
    nppiAlphaCompColorKey_8u_AC4R_Ctx
);

macro_rules! impl_alpha_premultiply_constant {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        /// Premultiplies source channels by the constant opacity `alpha` and writes
        /// the result to `destination`, using native scaling for the pixel type.
        /// Opacity ranges from zero to the maximum channel value.
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
            source: &ImageView<'_, $ty, $layout>,
            alpha: $ty,
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
                    alpha,
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

macro_rules! impl_alpha_premultiply_constant_in_place {
    ($name:ident, $ty:ty, $layout:ty, $ffi:ident) => {
        /// Premultiplies channels in `source_destination` by the constant opacity
        /// `alpha`, using native scaling for the pixel type.
        /// Opacity ranges from zero to the maximum channel value.
        ///
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the image exclusively writable through completion.
        /// Addresses, row strides, and image regions must satisfy the native routine's alignment
        /// and capacity requirements.
        /// Retain images and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            alpha: $ty,
            source_destination: &mut ImageViewMut<'_, $ty, $layout>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    alpha,
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

macro_rules! impl_alpha_premultiply {
    ($name:ident, $ty:ty, $ffi:ident) => {
        /// Premultiplies source channels by their pixel alpha values and writes
        /// the result to `destination`, using native scaling for the pixel type.
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
            source: &ImageView<'_, $ty, AC4>,
            destination: &mut ImageOutputView<'_, $ty, AC4>,
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

macro_rules! impl_alpha_premultiply_in_place {
    ($name:ident, $ty:ty, $ffi:ident) => {
        /// Premultiplies channels in `source_destination` by their pixel alpha values,
        /// using native scaling for the pixel type.
        ///
        #[doc = concat!("Calls [`sys::", stringify!($ffi), "`].")]
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails or an image belongs to another CUDA context.
        /// Native NPP failures are returned as errors. Success reports native submission.
        ///
        /// # Safety
        ///
        /// Input pixels must be initialized and ready when the stream reaches this operation.
        /// Keep the image exclusively writable through completion.
        /// Addresses, row strides, and image regions must satisfy the native routine's alignment
        /// and capacity requirements.
        /// Retain images and the stream context through queued completion,
        /// including submission errors and every captured graph replay.
        /// Padding and channels left untouched by the operation are not initialized by it.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $ty, AC4>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    source_destination.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u8_c1,
    u8,
    C1,
    nppiAlphaPremulC_8u_C1R_Ctx
);

impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u8_c3,
    u8,
    C3,
    nppiAlphaPremulC_8u_C3R_Ctx
);

impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u8_c4,
    u8,
    C4,
    nppiAlphaPremulC_8u_C4R_Ctx
);

impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u8_ac4,
    u8,
    AC4,
    nppiAlphaPremulC_8u_AC4R_Ctx
);

impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u16_c1,
    u16,
    C1,
    nppiAlphaPremulC_16u_C1R_Ctx
);

impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u16_c3,
    u16,
    C3,
    nppiAlphaPremulC_16u_C3R_Ctx
);

impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u16_c4,
    u16,
    C4,
    nppiAlphaPremulC_16u_C4R_Ctx
);

impl_alpha_premultiply_constant!(
    alpha_premultiply_constant_u16_ac4,
    u16,
    AC4,
    nppiAlphaPremulC_16u_AC4R_Ctx
);

impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u8_c1_in_place,
    u8,
    C1,
    nppiAlphaPremulC_8u_C1IR_Ctx
);

impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u8_c3_in_place,
    u8,
    C3,
    nppiAlphaPremulC_8u_C3IR_Ctx
);

impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u8_c4_in_place,
    u8,
    C4,
    nppiAlphaPremulC_8u_C4IR_Ctx
);

impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u8_ac4_in_place,
    u8,
    AC4,
    nppiAlphaPremulC_8u_AC4IR_Ctx
);

impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u16_c1_in_place,
    u16,
    C1,
    nppiAlphaPremulC_16u_C1IR_Ctx
);

impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u16_c3_in_place,
    u16,
    C3,
    nppiAlphaPremulC_16u_C3IR_Ctx
);

impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u16_c4_in_place,
    u16,
    C4,
    nppiAlphaPremulC_16u_C4IR_Ctx
);

impl_alpha_premultiply_constant_in_place!(
    alpha_premultiply_constant_u16_ac4_in_place,
    u16,
    AC4,
    nppiAlphaPremulC_16u_AC4IR_Ctx
);

impl_alpha_premultiply!(alpha_premultiply_u8_ac4, u8, nppiAlphaPremul_8u_AC4R_Ctx);

impl_alpha_premultiply!(alpha_premultiply_u16_ac4, u16, nppiAlphaPremul_16u_AC4R_Ctx);

impl_alpha_premultiply_in_place!(
    alpha_premultiply_u8_ac4_in_place,
    u8,
    nppiAlphaPremul_8u_AC4IR_Ctx
);

impl_alpha_premultiply_in_place!(
    alpha_premultiply_u16_ac4_in_place,
    u16,
    nppiAlphaPremul_16u_AC4IR_Ctx
);

impl_generic_alpha_comp_constant!("Composites `source1` and `source2` using the constant opacities `alpha1` and `alpha2` and the selected `operation`. Writes the result to `destination`.",

    AlphaCompConstantC1,
    alpha_comp_constant,
    alpha_comp_constant_c1,
    C1,
    [
        u8 => alpha_comp_constant_u8_c1,
        i8 => alpha_comp_constant_i8_c1,
        u16 => alpha_comp_constant_u16_c1,
        i16 => alpha_comp_constant_i16_c1,
        u32 => alpha_comp_constant_u32_c1,
        i32 => alpha_comp_constant_i32_c1,
        f32 => alpha_comp_constant_f32_c1,
    ]
);

impl_generic_alpha_comp_constant!("Composites `source1` and `source2` using the constant opacities `alpha1` and `alpha2` and the selected `operation`. Writes the result to `destination`.",

    AlphaCompConstantC3,
    alpha_comp_constant,
    alpha_comp_constant_c3,
    C3,
    [
        u8 => alpha_comp_constant_u8_c3,
        u16 => alpha_comp_constant_u16_c3,
    ]
);

impl_generic_alpha_comp_constant!("Composites `source1` and `source2` using the constant opacities `alpha1` and `alpha2` and the selected `operation`. Writes the result to `destination`.",

    AlphaCompConstantC4,
    alpha_comp_constant,
    alpha_comp_constant_c4,
    C4,
    [
        u8 => alpha_comp_constant_u8_c4,
        u16 => alpha_comp_constant_u16_c4,
    ]
);

impl_generic_alpha_comp_constant!("Composites `source1` and `source2` using the constant opacities `alpha1` and `alpha2` and the selected `operation`. Writes the result to `destination`.",

    AlphaCompConstantAc4,
    alpha_comp_constant,
    alpha_comp_constant_ac4,
    AC4,
    [
        u8 => alpha_comp_constant_u8_ac4,
        u16 => alpha_comp_constant_u16_ac4,
    ]
);

impl_generic_alpha_comp!("Composites `source1` and `source2` using their pixel alpha values and the selected `operation`. Writes the result to `destination`.",

    AlphaCompAc1,
    alpha_comp,
    alpha_comp_ac1,
    C1,
    [
        u8 => alpha_comp_u8_ac1,
        i8 => alpha_comp_i8_ac1,
        u16 => alpha_comp_u16_ac1,
        i16 => alpha_comp_i16_ac1,
        u32 => alpha_comp_u32_ac1,
        i32 => alpha_comp_i32_ac1,
        f32 => alpha_comp_f32_ac1,
    ]
);

impl_generic_alpha_comp!("Composites `source1` and `source2` using their pixel alpha values and the selected `operation`. Writes the result to `destination`.",

    AlphaCompAc4,
    alpha_comp,
    alpha_comp_ac4,
    AC4,
    [
        u8 => alpha_comp_u8_ac4,
        u16 => alpha_comp_u16_ac4,
        u32 => alpha_comp_u32_ac4,
        i32 => alpha_comp_i32_ac4,
        f32 => alpha_comp_f32_ac4,
    ]
);

impl_generic_constant_scalar_operation!("Premultiplies each processed channel by the opacity in `constant`, using native scaling for the pixel type. Writes the result to `destination`.",

    AlphaPremultiplyConstantC1,
    alpha_premultiply_constant,
    alpha_premultiply_constant_c1,
    C1,
    [
        u8, u8 => alpha_premultiply_constant_u8_c1,
        u16, u16 => alpha_premultiply_constant_u16_c1,
    ]
);

impl_generic_constant_scalar_operation_in_place!("Premultiplies each processed channel by the opacity in `constant`, using native scaling for the pixel type. Updates `source_destination` in place.",

    AlphaPremultiplyConstantC1InPlace,
    alpha_premultiply_constant_in_place,
    alpha_premultiply_constant_c1_in_place,
    C1,
    [
        u8, u8 => alpha_premultiply_constant_u8_c1_in_place,
        u16, u16 => alpha_premultiply_constant_u16_c1_in_place,
    ]
);

impl_generic_constant_scalar_operation!("Premultiplies each processed channel by the opacity in `constant`, using native scaling for the pixel type. Writes the result to `destination`.",

    AlphaPremultiplyConstantC3,
    alpha_premultiply_constant,
    alpha_premultiply_constant_c3,
    C3,
    [
        u8, u8 => alpha_premultiply_constant_u8_c3,
        u16, u16 => alpha_premultiply_constant_u16_c3,
    ]
);

impl_generic_constant_scalar_operation_in_place!("Premultiplies each processed channel by the opacity in `constant`, using native scaling for the pixel type. Updates `source_destination` in place.",

    AlphaPremultiplyConstantC3InPlace,
    alpha_premultiply_constant_in_place,
    alpha_premultiply_constant_c3_in_place,
    C3,
    [
        u8, u8 => alpha_premultiply_constant_u8_c3_in_place,
        u16, u16 => alpha_premultiply_constant_u16_c3_in_place,
    ]
);

impl_generic_constant_scalar_operation!("Premultiplies each processed channel by the opacity in `constant`, using native scaling for the pixel type. Writes the result to `destination`.",

    AlphaPremultiplyConstantC4,
    alpha_premultiply_constant,
    alpha_premultiply_constant_c4,
    C4,
    [
        u8, u8 => alpha_premultiply_constant_u8_c4,
        u16, u16 => alpha_premultiply_constant_u16_c4,
    ]
);

impl_generic_constant_scalar_operation_in_place!("Premultiplies each processed channel by the opacity in `constant`, using native scaling for the pixel type. Updates `source_destination` in place.",

    AlphaPremultiplyConstantC4InPlace,
    alpha_premultiply_constant_in_place,
    alpha_premultiply_constant_c4_in_place,
    C4,
    [
        u8, u8 => alpha_premultiply_constant_u8_c4_in_place,
        u16, u16 => alpha_premultiply_constant_u16_c4_in_place,
    ]
);

impl_generic_constant_scalar_operation!("Premultiplies each processed channel by the opacity in `constant`, using native scaling for the pixel type. Writes the result to `destination`.",

    AlphaPremultiplyConstantAc4,
    alpha_premultiply_constant,
    alpha_premultiply_constant_ac4,
    AC4,
    [
        u8, u8 => alpha_premultiply_constant_u8_ac4,
        u16, u16 => alpha_premultiply_constant_u16_ac4,
    ]
);

impl_generic_constant_scalar_operation_in_place!("Premultiplies each processed channel by the opacity in `constant`, using native scaling for the pixel type. Updates `source_destination` in place.",

    AlphaPremultiplyConstantAc4InPlace,
    alpha_premultiply_constant_in_place,
    alpha_premultiply_constant_ac4_in_place,
    AC4,
    [
        u8, u8 => alpha_premultiply_constant_u8_ac4_in_place,
        u16, u16 => alpha_premultiply_constant_u16_ac4_in_place,
    ]
);

impl_generic_unary_operation!("Premultiplies color channels by the alpha value stored in each source pixel, using native scaling for the pixel type. Writes the result to `destination`.",

    AlphaPremultiplyAc4,
    alpha_premultiply,
    alpha_premultiply_ac4,
    AC4,
    [
        u8 => alpha_premultiply_u8_ac4,
        u16 => alpha_premultiply_u16_ac4,
    ]
);

impl_generic_unary_operation_in_place!("Premultiplies color channels by the alpha value stored in each source pixel, using native scaling for the pixel type. Updates `source_destination` in place.",

    AlphaPremultiplyAc4InPlace,
    alpha_premultiply_in_place,
    alpha_premultiply_ac4_in_place,
    AC4,
    [
        u8 => alpha_premultiply_u8_ac4_in_place,
        u16 => alpha_premultiply_u16_ac4_in_place,
    ]
);
