use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering_flood_fill::direct::{
            flood_fill_gradient_boundary_u8_c1_in_place,
            flood_fill_gradient_boundary_u8_c3_in_place,
            flood_fill_gradient_boundary_u16_c1_in_place,
            flood_fill_gradient_boundary_u16_c3_in_place,
            flood_fill_gradient_boundary_u32_c1_in_place,
            flood_fill_gradient_boundary_u32_c3_in_place, flood_fill_gradient_u8_c1_in_place,
            flood_fill_gradient_u8_c3_in_place, flood_fill_gradient_u16_c1_in_place,
            flood_fill_gradient_u16_c3_in_place, flood_fill_gradient_u32_c1_in_place,
            flood_fill_gradient_u32_c3_in_place,
        },
        view::{C1, C3, ImageViewMut},
    },
    types::{ConnectedRegion, DataTypeLike, ImageNormalization, Point},
};

/// Dispatches this flood-fill operation for a native pixel element type.
///
/// # Safety
///
/// Preserve the native element representation, channel layout, connectivity semantics, and documented raw image execution contract.
pub unsafe trait FloodFillGradientC1InPlace: DataTypeLike {
    /// Flood-fills a seed-connected single-channel region in place.
    /// Eligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.
    /// Replaces eligible pixels with `new_value`.
    /// The wrapper does not validate range bounds or seed-relative tolerances.
    ///
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
    unsafe fn flood_fill_gradient_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        seed: Point,
        min: Self,
        max: Self,
        new_value: Self,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;
}

macro_rules! impl_flood_fill_gradient_c1_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        unsafe impl FloodFillGradientC1InPlace for $pixel_ty {
            unsafe fn flood_fill_gradient_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, C1>,
                seed: Point,
                min: Self,
                max: Self,
                new_value: Self,
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source_destination,
                        seed,
                        min,
                        max,
                        new_value,
                        norm,
                        connected_region,
                    )
                }
            }
        }
    };
}

impl_flood_fill_gradient_c1_dispatch!(u8, flood_fill_gradient_u8_c1_in_place);

impl_flood_fill_gradient_c1_dispatch!(u16, flood_fill_gradient_u16_c1_in_place);

impl_flood_fill_gradient_c1_dispatch!(u32, flood_fill_gradient_u32_c1_in_place);

/// Flood-fills a seed-connected single-channel region in place.
/// Eligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.
/// Replaces eligible pixels with `new_value`.
/// The wrapper does not validate range bounds or seed-relative tolerances.
///
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
pub unsafe fn flood_fill_gradient_in_place<T: FloodFillGradientC1InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C1>,
    seed: Point,
    min: T,
    max: T,
    new_value: T,
    norm: ImageNormalization,
    connected_region: Option<&mut ConnectedRegion>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::flood_fill_gradient_in_place(
            stream_context,
            source_destination,
            seed,
            min,
            max,
            new_value,
            norm,
            connected_region,
        )
    }
}

/// Dispatches this flood-fill operation for a native pixel element type.
///
/// # Safety
///
/// Preserve the native element representation, channel layout, connectivity semantics, and documented raw image execution contract.
pub unsafe trait FloodFillGradientC3InPlace: DataTypeLike {
    /// Flood-fills a seed-connected three-channel region in place.
    /// Eligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.
    /// Comparisons apply to all three components.
    /// Replaces eligible pixels with `new_values`.
    /// The wrapper does not validate range bounds or seed-relative tolerances.
    ///
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
    unsafe fn flood_fill_gradient_color_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C3>,
        seed: Point,
        min: [Self; 3],
        max: [Self; 3],
        new_values: [Self; 3],
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;
}

macro_rules! impl_flood_fill_gradient_c3_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        unsafe impl FloodFillGradientC3InPlace for $pixel_ty {
            unsafe fn flood_fill_gradient_color_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, C3>,
                seed: Point,
                min: [Self; 3],
                max: [Self; 3],
                new_values: [Self; 3],
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source_destination,
                        seed,
                        min,
                        max,
                        new_values,
                        norm,
                        connected_region,
                    )
                }
            }
        }
    };
}

impl_flood_fill_gradient_c3_dispatch!(u8, flood_fill_gradient_u8_c3_in_place);

impl_flood_fill_gradient_c3_dispatch!(u16, flood_fill_gradient_u16_c3_in_place);

impl_flood_fill_gradient_c3_dispatch!(u32, flood_fill_gradient_u32_c3_in_place);

/// Flood-fills a seed-connected three-channel region in place.
/// Eligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.
/// Comparisons apply to all three components.
/// Replaces eligible pixels with `new_values`.
/// The wrapper does not validate range bounds or seed-relative tolerances.
///
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
pub unsafe fn flood_fill_gradient_color_in_place<T: FloodFillGradientC3InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C3>,
    seed: Point,
    min: [T; 3],
    max: [T; 3],
    new_values: [T; 3],
    norm: ImageNormalization,
    connected_region: Option<&mut ConnectedRegion>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::flood_fill_gradient_color_in_place(
            stream_context,
            source_destination,
            seed,
            min,
            max,
            new_values,
            norm,
            connected_region,
        )
    }
}

/// Dispatches this flood-fill operation for a native pixel element type.
///
/// # Safety
///
/// Preserve the native element representation, channel layout, connectivity semantics, and documented raw image execution contract.
pub unsafe trait FloodFillGradientBoundaryC1InPlace: DataTypeLike {
    /// Flood-fills a seed-connected single-channel region in place.
    /// Eligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.
    /// Replaces eligible pixels with `new_value`.
    /// Colors the filled region's boundary with `boundary_value`.
    /// The wrapper does not validate range bounds or seed-relative tolerances.
    ///
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
    unsafe fn flood_fill_gradient_boundary_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C1>,
        seed: Point,
        min: Self,
        max: Self,
        new_value: Self,
        boundary_value: Self,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;
}

macro_rules! impl_flood_fill_gradient_boundary_c1_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        unsafe impl FloodFillGradientBoundaryC1InPlace for $pixel_ty {
            unsafe fn flood_fill_gradient_boundary_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, C1>,
                seed: Point,
                min: Self,
                max: Self,
                new_value: Self,
                boundary_value: Self,
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source_destination,
                        seed,
                        min,
                        max,
                        new_value,
                        boundary_value,
                        norm,
                        connected_region,
                    )
                }
            }
        }
    };
}

impl_flood_fill_gradient_boundary_c1_dispatch!(u8, flood_fill_gradient_boundary_u8_c1_in_place);

impl_flood_fill_gradient_boundary_c1_dispatch!(u16, flood_fill_gradient_boundary_u16_c1_in_place);

impl_flood_fill_gradient_boundary_c1_dispatch!(u32, flood_fill_gradient_boundary_u32_c1_in_place);

/// Flood-fills a seed-connected single-channel region in place.
/// Eligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.
/// Replaces eligible pixels with `new_value`.
/// Colors the filled region's boundary with `boundary_value`.
/// The wrapper does not validate range bounds or seed-relative tolerances.
///
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
pub unsafe fn flood_fill_gradient_boundary_in_place<T: FloodFillGradientBoundaryC1InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C1>,
    seed: Point,
    min: T,
    max: T,
    new_value: T,
    boundary_value: T,
    norm: ImageNormalization,
    connected_region: Option<&mut ConnectedRegion>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::flood_fill_gradient_boundary_in_place(
            stream_context,
            source_destination,
            seed,
            min,
            max,
            new_value,
            boundary_value,
            norm,
            connected_region,
        )
    }
}

/// Dispatches this flood-fill operation for a native pixel element type.
///
/// # Safety
///
/// Preserve the native element representation, channel layout, connectivity semantics, and documented raw image execution contract.
pub unsafe trait FloodFillGradientBoundaryC3InPlace: DataTypeLike {
    /// Flood-fills a seed-connected three-channel region in place.
    /// Eligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.
    /// Comparisons apply to all three components.
    /// Replaces eligible pixels with `new_values`.
    /// Colors the filled region's boundary with `boundary_values`.
    /// The wrapper does not validate range bounds or seed-relative tolerances.
    ///
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
    unsafe fn flood_fill_gradient_boundary_color_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, Self, C3>,
        seed: Point,
        min: [Self; 3],
        max: [Self; 3],
        new_values: [Self; 3],
        boundary_values: [Self; 3],
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
    ) -> Result<()>;
}

macro_rules! impl_flood_fill_gradient_boundary_c3_dispatch {
    ($pixel_ty:ty, $name:ident) => {
        unsafe impl FloodFillGradientBoundaryC3InPlace for $pixel_ty {
            unsafe fn flood_fill_gradient_boundary_color_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, Self, C3>,
                seed: Point,
                min: [Self; 3],
                max: [Self; 3],
                new_values: [Self; 3],
                boundary_values: [Self; 3],
                norm: ImageNormalization,
                connected_region: Option<&mut ConnectedRegion>,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source_destination,
                        seed,
                        min,
                        max,
                        new_values,
                        boundary_values,
                        norm,
                        connected_region,
                    )
                }
            }
        }
    };
}

impl_flood_fill_gradient_boundary_c3_dispatch!(u8, flood_fill_gradient_boundary_u8_c3_in_place);

impl_flood_fill_gradient_boundary_c3_dispatch!(u16, flood_fill_gradient_boundary_u16_c3_in_place);

impl_flood_fill_gradient_boundary_c3_dispatch!(u32, flood_fill_gradient_boundary_u32_c3_in_place);

/// Flood-fills a seed-connected three-channel region in place.
/// Eligible original values lie between the original seed value minus `min` and the original seed value plus `max`, inclusive.
/// Comparisons apply to all three components.
/// Replaces eligible pixels with `new_values`.
/// Colors the filled region's boundary with `boundary_values`.
/// The wrapper does not validate range bounds or seed-relative tolerances.
///
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
pub unsafe fn flood_fill_gradient_boundary_color_in_place<T: FloodFillGradientBoundaryC3InPlace>(
    stream_context: &StreamContext,
    source_destination: &mut ImageViewMut<'_, T, C3>,
    seed: Point,
    min: [T; 3],
    max: [T; 3],
    new_values: [T; 3],
    boundary_values: [T; 3],
    norm: ImageNormalization,
    connected_region: Option<&mut ConnectedRegion>,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::flood_fill_gradient_boundary_color_in_place(
            stream_context,
            source_destination,
            seed,
            min,
            max,
            new_values,
            boundary_values,
            norm,
            connected_region,
        )
    }
}
