use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_validation::{validate_positive_size, validate_seed},
        view::{C1, C3, ImageViewMut},
    },
    try_ffi,
    types::{ConnectedRegion, ImageNormalization, Point, Size},
    utility::to_usize,
};

/// Returns the device scratch size in bytes for flood filling an image of size `roi`.
///
/// # Errors
///
/// Returns an error if either ROI dimension is nonpositive, NPP rejects the size query, or the returned size cannot be represented as `usize`.
pub fn flood_fill_buffer_size(roi: Size) -> Result<usize> {
    validate_positive_size(roi)?;
    let mut bytes = 0;
    // SAFETY: `nppiFloodFillGetBufferSize` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiFloodFillGetBufferSize(roi.into(), &raw mut bytes,))?;
    }
    to_usize(bytes, "buffer size")
}

macro_rules! flood_fill_region_ptr {
    ($region:expr) => {
        match $region {
            Some(region) => &raw mut *region,
            None => std::ptr::null_mut(),
        }
    };
}

macro_rules! impl_flood_fill_c1 {
    ($doc:literal, $name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        /// `ImageNormalization::L1` selects four-way connectivity; `ImageNormalization::Infinity` selects eight-way connectivity.
        /// The wrapper forwards `norm` without restricting it to these modes.
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// `Some(connected_region)` receives region information in host memory; `None` disables that output.
        /// The region bounding box stores left, top, right, and bottom coordinates in its `x`, `y`, `width`, and `height` fields.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the image belongs to another CUDA context, the image size is nonpositive, or `seed` lies outside the image.
        /// Capture checks, scratch-size queries, device allocation, native execution, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image values must be initialized and ready for native execution.
        /// Keep the image exclusively accessible during native use and retain its allocation until this call completes its native work.
        /// Element representation, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C1>,
            seed: Point,
            new_value: $pixel_ty,
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_seed(source_destination.size(), seed)?;
            let scratch = DeviceMemory::<u8>::create_uninit(
                stream_context.cuda_context(),
                flood_fill_buffer_size(source_destination.size())?,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    new_value,
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_flood_fill_c3 {
    ($doc:literal, $name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        /// `ImageNormalization::L1` selects four-way connectivity; `ImageNormalization::Infinity` selects eight-way connectivity.
        /// The wrapper forwards `norm` without restricting it to these modes.
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// `Some(connected_region)` receives region information in host memory; `None` disables that output.
        /// The region bounding box stores left, top, right, and bottom coordinates in its `x`, `y`, `width`, and `height` fields.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the image belongs to another CUDA context, the image size is nonpositive, or `seed` lies outside the image.
        /// Capture checks, scratch-size queries, device allocation, native execution, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image values must be initialized and ready for native execution.
        /// Keep the image exclusively accessible during native use and retain its allocation until this call completes its native work.
        /// Element representation, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C3>,
            seed: Point,
            new_values: [$pixel_ty; 3],
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_seed(source_destination.size(), seed)?;
            let scratch = DeviceMemory::<u8>::create_uninit(
                stream_context.cuda_context(),
                flood_fill_buffer_size(source_destination.size())?,
            )?;

            // SAFETY: the `ffi_name` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    new_values.as_ptr().cast(),
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_flood_fill_boundary_c1 {
    ($doc:literal, $name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        /// `ImageNormalization::L1` selects four-way connectivity; `ImageNormalization::Infinity` selects eight-way connectivity.
        /// The wrapper forwards `norm` without restricting it to these modes.
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// `Some(connected_region)` receives region information in host memory; `None` disables that output.
        /// The region bounding box stores left, top, right, and bottom coordinates in its `x`, `y`, `width`, and `height` fields.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the image belongs to another CUDA context, the image size is nonpositive, or `seed` lies outside the image.
        /// Capture checks, scratch-size queries, device allocation, native execution, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image values must be initialized and ready for native execution.
        /// Keep the image exclusively accessible during native use and retain its allocation until this call completes its native work.
        /// Element representation, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C1>,
            seed: Point,
            new_value: $pixel_ty,
            boundary_value: $pixel_ty,
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_seed(source_destination.size(), seed)?;
            let scratch = DeviceMemory::<u8>::create_uninit(
                stream_context.cuda_context(),
                flood_fill_buffer_size(source_destination.size())?,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    new_value,
                    boundary_value,
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_flood_fill_boundary_c3 {
    ($doc:literal, $name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        /// `ImageNormalization::L1` selects four-way connectivity; `ImageNormalization::Infinity` selects eight-way connectivity.
        /// The wrapper forwards `norm` without restricting it to these modes.
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// `Some(connected_region)` receives region information in host memory; `None` disables that output.
        /// The region bounding box stores left, top, right, and bottom coordinates in its `x`, `y`, `width`, and `height` fields.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the image belongs to another CUDA context, the image size is nonpositive, or `seed` lies outside the image.
        /// Capture checks, scratch-size queries, device allocation, native execution, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image values must be initialized and ready for native execution.
        /// Keep the image exclusively accessible during native use and retain its allocation until this call completes its native work.
        /// Element representation, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C3>,
            seed: Point,
            new_values: [$pixel_ty; 3],
            boundary_values: [$pixel_ty; 3],
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_seed(source_destination.size(), seed)?;
            let scratch = DeviceMemory::<u8>::create_uninit(
                stream_context.cuda_context(),
                flood_fill_buffer_size(source_destination.size())?,
            )?;

            // SAFETY: the `ffi_name` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    new_values.as_ptr().cast(),
                    boundary_values.as_ptr().cast(),
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_flood_fill_range_c1 {
    ($doc:literal, $name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        /// `ImageNormalization::L1` selects four-way connectivity; `ImageNormalization::Infinity` selects eight-way connectivity.
        /// The wrapper forwards `norm` without restricting it to these modes.
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// `Some(connected_region)` receives region information in host memory; `None` disables that output.
        /// The region bounding box stores left, top, right, and bottom coordinates in its `x`, `y`, `width`, and `height` fields.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the image belongs to another CUDA context, the image size is nonpositive, or `seed` lies outside the image.
        /// Capture checks, scratch-size queries, device allocation, native execution, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image values must be initialized and ready for native execution.
        /// Keep the image exclusively accessible during native use and retain its allocation until this call completes its native work.
        /// Element representation, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C1>,
            seed: Point,
            min: $pixel_ty,
            max: $pixel_ty,
            new_value: $pixel_ty,
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_seed(source_destination.size(), seed)?;
            let scratch = DeviceMemory::<u8>::create_uninit(
                stream_context.cuda_context(),
                flood_fill_buffer_size(source_destination.size())?,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    min,
                    max,
                    new_value,
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_flood_fill_range_c3 {
    ($doc:literal, $name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        /// `ImageNormalization::L1` selects four-way connectivity; `ImageNormalization::Infinity` selects eight-way connectivity.
        /// The wrapper forwards `norm` without restricting it to these modes.
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// `Some(connected_region)` receives region information in host memory; `None` disables that output.
        /// The region bounding box stores left, top, right, and bottom coordinates in its `x`, `y`, `width`, and `height` fields.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the image belongs to another CUDA context, the image size is nonpositive, or `seed` lies outside the image.
        /// Capture checks, scratch-size queries, device allocation, native execution, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image values must be initialized and ready for native execution.
        /// Keep the image exclusively accessible during native use and retain its allocation until this call completes its native work.
        /// Element representation, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C3>,
            seed: Point,
            min: [$pixel_ty; 3],
            max: [$pixel_ty; 3],
            new_values: [$pixel_ty; 3],
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_seed(source_destination.size(), seed)?;
            let mut min = min;
            let mut max = max;
            let scratch = DeviceMemory::<u8>::create_uninit(
                stream_context.cuda_context(),
                flood_fill_buffer_size(source_destination.size())?,
            )?;

            // SAFETY: the `ffi_name` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    min.as_mut_ptr().cast(),
                    max.as_mut_ptr().cast(),
                    new_values.as_ptr().cast(),
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_flood_fill_range_boundary_c1 {
    ($doc:literal, $name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        /// `ImageNormalization::L1` selects four-way connectivity; `ImageNormalization::Infinity` selects eight-way connectivity.
        /// The wrapper forwards `norm` without restricting it to these modes.
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// `Some(connected_region)` receives region information in host memory; `None` disables that output.
        /// The region bounding box stores left, top, right, and bottom coordinates in its `x`, `y`, `width`, and `height` fields.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the image belongs to another CUDA context, the image size is nonpositive, or `seed` lies outside the image.
        /// Capture checks, scratch-size queries, device allocation, native execution, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image values must be initialized and ready for native execution.
        /// Keep the image exclusively accessible during native use and retain its allocation until this call completes its native work.
        /// Element representation, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C1>,
            seed: Point,
            min: $pixel_ty,
            max: $pixel_ty,
            new_value: $pixel_ty,
            boundary_value: $pixel_ty,
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_seed(source_destination.size(), seed)?;
            let scratch = DeviceMemory::<u8>::create_uninit(
                stream_context.cuda_context(),
                flood_fill_buffer_size(source_destination.size())?,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    min,
                    max,
                    new_value,
                    boundary_value,
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_flood_fill_range_boundary_c3 {
    ($doc:literal, $name:ident, $pixel_ty:ty, $ffi_name:ident) => {
        #[doc = $doc]
        #[doc = concat!("Calls [`sys::", stringify!($ffi_name), "`].")]
        /// `ImageNormalization::L1` selects four-way connectivity; `ImageNormalization::Infinity` selects eight-way connectivity.
        /// The wrapper forwards `norm` without restricting it to these modes.
        /// Owns temporary device scratch, rejects stream capture, and returns after successful stream completion.
        /// `Some(connected_region)` receives region information in host memory; `None` disables that output.
        /// The region bounding box stores left, top, right, and bottom coordinates in its `x`, `y`, `width`, and `height` fields.
        /// Row padding remains unwritten.
        ///
        /// # Errors
        ///
        /// Returns an error if CUDA context binding fails, the image belongs to another CUDA context, the image size is nonpositive, or `seed` lies outside the image.
        /// Capture checks, scratch-size queries, device allocation, native execution, and stream synchronization can also fail.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Image values must be initialized and ready for native execution.
        /// Keep the image exclusively accessible during native use and retain its allocation until this call completes its native work.
        /// Element representation, byte strides, alignment, and allocation bounds must satisfy the native routine.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source_destination: &mut ImageViewMut<'_, $pixel_ty, C3>,
            seed: Point,
            min: [$pixel_ty; 3],
            max: [$pixel_ty; 3],
            new_values: [$pixel_ty; 3],
            boundary_values: [$pixel_ty; 3],
            norm: ImageNormalization,
            connected_region: Option<&mut ConnectedRegion>,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(source_destination)?;
            stream_context.stream().ensure_not_capturing()?;

            validate_seed(source_destination.size(), seed)?;
            let mut min = min;
            let mut max = max;
            let scratch = DeviceMemory::<u8>::create_uninit(
                stream_context.cuda_context(),
                flood_fill_buffer_size(source_destination.size())?,
            )?;

            // SAFETY: the `ffi_name` entry point receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi_name(
                    source_destination.as_mut_ptr().cast(),
                    source_destination.step(),
                    seed.into(),
                    min.as_mut_ptr().cast(),
                    max.as_mut_ptr().cast(),
                    new_values.as_ptr().cast(),
                    boundary_values.as_ptr().cast(),
                    norm.into(),
                    source_destination.size().into(),
                    flood_fill_region_ptr!(connected_region).cast(),
                    scratch.as_mut_ptr().cast(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

impl_flood_fill_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible pixels match the original seed pixel value.\nReplaces eligible pixels with `new_value`.",
    flood_fill_u8_c1_in_place,
    u8,
    nppiFloodFill_8u_C1IR_Ctx
);

impl_flood_fill_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible pixels match the original seed pixel value.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.",
    flood_fill_u8_c3_in_place,
    u8,
    nppiFloodFill_8u_C3IR_Ctx
);

impl_flood_fill_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible pixels match the original seed pixel value.\nReplaces eligible pixels with `new_value`.",
    flood_fill_u16_c1_in_place,
    u16,
    nppiFloodFill_16u_C1IR_Ctx
);

impl_flood_fill_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible pixels match the original seed pixel value.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.",
    flood_fill_u16_c3_in_place,
    u16,
    nppiFloodFill_16u_C3IR_Ctx
);

impl_flood_fill_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible pixels match the original seed pixel value.\nReplaces eligible pixels with `new_value`.",
    flood_fill_u32_c1_in_place,
    u32,
    nppiFloodFill_32u_C1IR_Ctx
);

impl_flood_fill_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible pixels match the original seed pixel value.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.",
    flood_fill_u32_c3_in_place,
    u32,
    nppiFloodFill_32u_C3IR_Ctx
);

impl_flood_fill_boundary_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible pixels match the original seed pixel value.\nReplaces eligible pixels with `new_value`.\nColors the filled region's boundary with `boundary_value`.",
    flood_fill_boundary_u8_c1_in_place,
    u8,
    nppiFloodFillBoundary_8u_C1IR_Ctx
);

impl_flood_fill_boundary_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible pixels match the original seed pixel value.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nColors the filled region's boundary with `boundary_values`.",
    flood_fill_boundary_u8_c3_in_place,
    u8,
    nppiFloodFillBoundary_8u_C3IR_Ctx
);

impl_flood_fill_boundary_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible pixels match the original seed pixel value.\nReplaces eligible pixels with `new_value`.\nColors the filled region's boundary with `boundary_value`.",
    flood_fill_boundary_u16_c1_in_place,
    u16,
    nppiFloodFillBoundary_16u_C1IR_Ctx
);

impl_flood_fill_boundary_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible pixels match the original seed pixel value.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nColors the filled region's boundary with `boundary_values`.",
    flood_fill_boundary_u16_c3_in_place,
    u16,
    nppiFloodFillBoundary_16u_C3IR_Ctx
);

impl_flood_fill_boundary_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible pixels match the original seed pixel value.\nReplaces eligible pixels with `new_value`.\nColors the filled region's boundary with `boundary_value`.",
    flood_fill_boundary_u32_c1_in_place,
    u32,
    nppiFloodFillBoundary_32u_C1IR_Ctx
);

impl_flood_fill_boundary_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible pixels match the original seed pixel value.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nColors the filled region's boundary with `boundary_values`.",
    flood_fill_boundary_u32_c3_in_place,
    u32,
    nppiFloodFillBoundary_32u_C3IR_Ctx
);

impl_flood_fill_range_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible original values lie in the inclusive range `min..=max`.\nReplaces eligible pixels with `new_value`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_range_u8_c1_in_place,
    u8,
    nppiFloodFillRange_8u_C1IR_Ctx
);

impl_flood_fill_range_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible original values lie in the inclusive range `min..=max`.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_range_u8_c3_in_place,
    u8,
    nppiFloodFillRange_8u_C3IR_Ctx
);

impl_flood_fill_range_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible original values lie in the inclusive range `min..=max`.\nReplaces eligible pixels with `new_value`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_range_u16_c1_in_place,
    u16,
    nppiFloodFillRange_16u_C1IR_Ctx
);

impl_flood_fill_range_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible original values lie in the inclusive range `min..=max`.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_range_u16_c3_in_place,
    u16,
    nppiFloodFillRange_16u_C3IR_Ctx
);

impl_flood_fill_range_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible original values lie in the inclusive range `min..=max`.\nReplaces eligible pixels with `new_value`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_range_u32_c1_in_place,
    u32,
    nppiFloodFillRange_32u_C1IR_Ctx
);

impl_flood_fill_range_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible original values lie in the inclusive range `min..=max`.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_range_u32_c3_in_place,
    u32,
    nppiFloodFillRange_32u_C3IR_Ctx
);

impl_flood_fill_range_boundary_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible original values lie in the inclusive range `min..=max`.\nReplaces eligible pixels with `new_value`.\nColors the filled region's boundary with `boundary_value`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_range_boundary_u8_c1_in_place,
    u8,
    nppiFloodFillRangeBoundary_8u_C1IR_Ctx
);

impl_flood_fill_range_boundary_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible original values lie in the inclusive range `min..=max`.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nColors the filled region's boundary with `boundary_values`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_range_boundary_u8_c3_in_place,
    u8,
    nppiFloodFillRangeBoundary_8u_C3IR_Ctx
);

impl_flood_fill_range_boundary_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible original values lie in the inclusive range `min..=max`.\nReplaces eligible pixels with `new_value`.\nColors the filled region's boundary with `boundary_value`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_range_boundary_u16_c1_in_place,
    u16,
    nppiFloodFillRangeBoundary_16u_C1IR_Ctx
);

impl_flood_fill_range_boundary_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible original values lie in the inclusive range `min..=max`.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nColors the filled region's boundary with `boundary_values`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_range_boundary_u16_c3_in_place,
    u16,
    nppiFloodFillRangeBoundary_16u_C3IR_Ctx
);

impl_flood_fill_range_boundary_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible original values lie in the inclusive range `min..=max`.\nReplaces eligible pixels with `new_value`.\nColors the filled region's boundary with `boundary_value`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_range_boundary_u32_c1_in_place,
    u32,
    nppiFloodFillRangeBoundary_32u_C1IR_Ctx
);

impl_flood_fill_range_boundary_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible original values lie in the inclusive range `min..=max`.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nColors the filled region's boundary with `boundary_values`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_range_boundary_u32_c3_in_place,
    u32,
    nppiFloodFillRangeBoundary_32u_C3IR_Ctx
);

impl_flood_fill_range_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.\nReplaces eligible pixels with `new_value`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_gradient_u8_c1_in_place,
    u8,
    nppiFloodFillGradient_8u_C1IR_Ctx
);

impl_flood_fill_range_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_gradient_u8_c3_in_place,
    u8,
    nppiFloodFillGradient_8u_C3IR_Ctx
);

impl_flood_fill_range_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.\nReplaces eligible pixels with `new_value`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_gradient_u16_c1_in_place,
    u16,
    nppiFloodFillGradient_16u_C1IR_Ctx
);

impl_flood_fill_range_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_gradient_u16_c3_in_place,
    u16,
    nppiFloodFillGradient_16u_C3IR_Ctx
);

impl_flood_fill_range_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.\nReplaces eligible pixels with `new_value`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_gradient_u32_c1_in_place,
    u32,
    nppiFloodFillGradient_32u_C1IR_Ctx
);

impl_flood_fill_range_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_gradient_u32_c3_in_place,
    u32,
    nppiFloodFillGradient_32u_C3IR_Ctx
);

impl_flood_fill_range_boundary_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.\nReplaces eligible pixels with `new_value`.\nColors the filled region's boundary with `boundary_value`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_gradient_boundary_u8_c1_in_place,
    u8,
    nppiFloodFillGradientBoundary_8u_C1IR_Ctx
);

impl_flood_fill_range_boundary_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nColors the filled region's boundary with `boundary_values`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_gradient_boundary_u8_c3_in_place,
    u8,
    nppiFloodFillGradientBoundary_8u_C3IR_Ctx
);

impl_flood_fill_range_boundary_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.\nReplaces eligible pixels with `new_value`.\nColors the filled region's boundary with `boundary_value`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_gradient_boundary_u16_c1_in_place,
    u16,
    nppiFloodFillGradientBoundary_16u_C1IR_Ctx
);

impl_flood_fill_range_boundary_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nColors the filled region's boundary with `boundary_values`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_gradient_boundary_u16_c3_in_place,
    u16,
    nppiFloodFillGradientBoundary_16u_C3IR_Ctx
);

impl_flood_fill_range_boundary_c1!(
    "Flood-fills a seed-connected single-channel region in place.\nEligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.\nReplaces eligible pixels with `new_value`.\nColors the filled region's boundary with `boundary_value`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_gradient_boundary_u32_c1_in_place,
    u32,
    nppiFloodFillGradientBoundary_32u_C1IR_Ctx
);

impl_flood_fill_range_boundary_c3!(
    "Flood-fills a seed-connected three-channel region in place.\nEligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.\nComparisons apply to all three components.\nReplaces eligible pixels with `new_values`.\nColors the filled region's boundary with `boundary_values`.\nThe wrapper does not validate range bounds or seed-relative tolerances.",
    flood_fill_gradient_boundary_u32_c3_in_place,
    u32,
    nppiFloodFillGradientBoundary_32u_C3IR_Ctx
);
