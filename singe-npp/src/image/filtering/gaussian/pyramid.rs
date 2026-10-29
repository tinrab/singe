use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        filtering_validation::{
            validate_advanced_filter_kernel, validate_gauss_pyramid_rate, validate_positive_size,
            validate_source_offset,
        },
        view::{C1, C3, ChannelLayout, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, Point, Size},
};

/// Returns the destination ROI for Gaussian-pyramid downsampling.
///
/// # Errors
///
/// Returns an error if either source ROI dimension is nonpositive, `rate` is not greater
/// than `1.0` and at most `10.0`, or NPP rejects the size query.
pub fn filter_gauss_pyramid_layer_down_border_dst_roi(source_roi: Size, rate: f32) -> Result<Size> {
    validate_positive_size(source_roi)?;
    validate_gauss_pyramid_rate(rate)?;
    let mut destination = sys::NppiSize {
        width: 0,
        height: 0,
    };
    // SAFETY: `nppiGetFilterGaussPyramidLayerDownBorderDstROI` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiGetFilterGaussPyramidLayerDownBorderDstROI(
            source_roi.width,
            source_roi.height,
            &raw mut destination,
            rate,
        ))?;
    }
    Ok(destination.into())
}

/// Returns the minimum and maximum destination ROIs for Gaussian-pyramid upsampling.
///
/// # Errors
///
/// Returns an error if either source ROI dimension is nonpositive, `rate` is not greater
/// than `1.0` and at most `10.0`, or NPP rejects the size query.
pub fn filter_gauss_pyramid_layer_up_border_dst_roi(
    source_roi: Size,
    rate: f32,
) -> Result<(Size, Size)> {
    validate_positive_size(source_roi)?;
    validate_gauss_pyramid_rate(rate)?;
    let mut destination_min = sys::NppiSize {
        width: 0,
        height: 0,
    };
    let mut destination_max = sys::NppiSize {
        width: 0,
        height: 0,
    };
    // SAFETY: `nppiGetFilterGaussPyramidLayerUpBorderDstROI` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiGetFilterGaussPyramidLayerUpBorderDstROI(
            source_roi.width,
            source_roi.height,
            &raw mut destination_min,
            &raw mut destination_max,
            rate,
        ))?;
    }
    Ok((destination_min.into(), destination_max.into()))
}

impl_filter_gauss_pyramid_border!(
    "Single channel 8-bit unsigned Gauss filter with downsampling and border control.",
    filter_gauss_pyramid_layer_down_border_u8_c1,
    u8,
    C1,
    nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx
);

impl_filter_gauss_pyramid_border!(
    "Three channel 8-bit unsigned Gauss filter with downsampling and border control.",
    filter_gauss_pyramid_layer_down_border_u8_c3,
    u8,
    C3,
    nppiFilterGaussPyramidLayerDownBorder_8u_C3R_Ctx
);

impl_filter_gauss_pyramid_border!(
    "Single channel 16-bit unsigned Gauss filter with downsampling and border control.",
    filter_gauss_pyramid_layer_down_border_u16_c1,
    u16,
    C1,
    nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx
);

impl_filter_gauss_pyramid_border!(
    "Three channel 16-bit unsigned Gauss filter with downsampling and border control.",
    filter_gauss_pyramid_layer_down_border_u16_c3,
    u16,
    C3,
    nppiFilterGaussPyramidLayerDownBorder_16u_C3R_Ctx
);

impl_filter_gauss_pyramid_border!(
    "Single channel 32-bit floating-point Gauss filter downsampling and with border control.",
    filter_gauss_pyramid_layer_down_border_f32_c1,
    f32,
    C1,
    nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx
);

impl_filter_gauss_pyramid_border!(
    "Three channel 32-bit floating-point Gauss filter with downsampling and border control.",
    filter_gauss_pyramid_layer_down_border_f32_c3,
    f32,
    C3,
    nppiFilterGaussPyramidLayerDownBorder_32f_C3R_Ctx
);

impl_filter_gauss_pyramid_border!(
    "Single channel 8-bit unsigned Gauss filter with upsampling and border control.",
    filter_gauss_pyramid_layer_up_border_u8_c1,
    u8,
    C1,
    nppiFilterGaussPyramidLayerUpBorder_8u_C1R_Ctx
);

impl_filter_gauss_pyramid_border!(
    "Three channel 8-bit unsigned Gauss filter with upsampling and border control.",
    filter_gauss_pyramid_layer_up_border_u8_c3,
    u8,
    C3,
    nppiFilterGaussPyramidLayerUpBorder_8u_C3R_Ctx
);

impl_filter_gauss_pyramid_border!(
    "Single channel 16-bit unsigned Gauss filter with upsampling and border control.",
    filter_gauss_pyramid_layer_up_border_u16_c1,
    u16,
    C1,
    nppiFilterGaussPyramidLayerUpBorder_16u_C1R_Ctx
);

impl_filter_gauss_pyramid_border!(
    "Three channel 16-bit unsigned Gauss filter with upsampling and border control.",
    filter_gauss_pyramid_layer_up_border_u16_c3,
    u16,
    C3,
    nppiFilterGaussPyramidLayerUpBorder_16u_C3R_Ctx
);

impl_filter_gauss_pyramid_border!(
    "Single channel 32-bit floating-point Gauss filter upsampling and with border control.",
    filter_gauss_pyramid_layer_up_border_f32_c1,
    f32,
    C1,
    nppiFilterGaussPyramidLayerUpBorder_32f_C1R_Ctx
);

impl_filter_gauss_pyramid_border!(
    "Three channel 32-bit floating-point Gauss filter with upsampling and border control.",
    filter_gauss_pyramid_layer_up_border_f32_c3,
    f32,
    C3,
    nppiFilterGaussPyramidLayerUpBorder_32f_C3R_Ctx
);

/// Dispatches Gaussian-pyramid filtering and downsampling with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel and layout representations, filter semantics,
/// and sampling requirements. Uphold the method's validation, source-halo requirements where
/// applicable, coefficient address-space requirements, and queued allocation lifetimes.
pub unsafe trait FilterGaussPyramidLayerDownBorder<L: ChannelLayout>: DataTypeLike {
    /// Filters and downsamples a Gaussian-pyramid layer with border control.
    ///
    /// Enqueues work on the selected stream.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// `source_offset` must address a source pixel, and `rate` must be greater than `1.0` and at most `10.0`.
    /// `kernel` must be nonempty and its length must fit in `i32`.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
    /// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn filter_gauss_pyramid_layer_down_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_gauss_pyramid_layer_down_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        unsafe impl FilterGaussPyramidLayerDownBorder<$layout> for $pixel_ty {
            unsafe fn filter_gauss_pyramid_layer_down_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                rate: f32,
                kernel: &[f32],
                border_type: BorderType,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        rate,
                        kernel,
                        border_type,
                    )
                }
            }
        }
    };
}

impl_filter_gauss_pyramid_layer_down_border_dispatch!(
    u8,
    C1,
    filter_gauss_pyramid_layer_down_border_u8_c1
);

impl_filter_gauss_pyramid_layer_down_border_dispatch!(
    u8,
    C3,
    filter_gauss_pyramid_layer_down_border_u8_c3
);

impl_filter_gauss_pyramid_layer_down_border_dispatch!(
    u16,
    C1,
    filter_gauss_pyramid_layer_down_border_u16_c1
);

impl_filter_gauss_pyramid_layer_down_border_dispatch!(
    u16,
    C3,
    filter_gauss_pyramid_layer_down_border_u16_c3
);

impl_filter_gauss_pyramid_layer_down_border_dispatch!(
    f32,
    C1,
    filter_gauss_pyramid_layer_down_border_f32_c1
);

impl_filter_gauss_pyramid_layer_down_border_dispatch!(
    f32,
    C3,
    filter_gauss_pyramid_layer_down_border_f32_c3
);

/// Filters and downsamples a Gaussian-pyramid layer with border control.
///
/// Enqueues work on the selected stream.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
/// `source_offset` must address a source pixel, and `rate` must be greater than `1.0` and at most `10.0`.
/// `kernel` must be nonempty and its length must fit in `i32`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
/// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn filter_gauss_pyramid_layer_down_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    rate: f32,
    kernel: &[f32],
    border_type: BorderType,
) -> Result<()>
where
    T: FilterGaussPyramidLayerDownBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_gauss_pyramid_layer_down_border(
            stream_context,
            source,
            source_offset,
            destination,
            rate,
            kernel,
            border_type,
        )
    }
}

/// Dispatches Gaussian-pyramid filtering and upsampling with border control.
///
/// # Safety
///
/// Implementations must preserve native pixel and layout representations, filter semantics,
/// and sampling requirements. Uphold the method's validation, source-halo requirements where
/// applicable, coefficient address-space requirements, and queued allocation lifetimes.
pub unsafe trait FilterGaussPyramidLayerUpBorder<L: ChannelLayout>: DataTypeLike {
    /// Filters and upsamples a Gaussian-pyramid layer with border control.
    ///
    /// Enqueues work on the selected stream.
    /// Ignored channels and row padding may remain unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
    /// `source_offset` must address a source pixel, and `rate` must be greater than `1.0` and at most `10.0`.
    /// `kernel` must be nonempty and its length must fit in `i32`.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Source samples must be initialized and ready when read by the selected stream.
    /// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
    /// Retain image allocations through queued completion, including submission errors and captured graph replay.
    /// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
    /// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
    unsafe fn filter_gauss_pyramid_layer_up_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, Self, L>,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_filter_gauss_pyramid_layer_up_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        unsafe impl FilterGaussPyramidLayerUpBorder<$layout> for $pixel_ty {
            unsafe fn filter_gauss_pyramid_layer_up_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, Self, $layout>,
                rate: f32,
                kernel: &[f32],
                border_type: BorderType,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        rate,
                        kernel,
                        border_type,
                    )
                }
            }
        }
    };
}

impl_filter_gauss_pyramid_layer_up_border_dispatch!(
    u8,
    C1,
    filter_gauss_pyramid_layer_up_border_u8_c1
);

impl_filter_gauss_pyramid_layer_up_border_dispatch!(
    u8,
    C3,
    filter_gauss_pyramid_layer_up_border_u8_c3
);

impl_filter_gauss_pyramid_layer_up_border_dispatch!(
    u16,
    C1,
    filter_gauss_pyramid_layer_up_border_u16_c1
);

impl_filter_gauss_pyramid_layer_up_border_dispatch!(
    u16,
    C3,
    filter_gauss_pyramid_layer_up_border_u16_c3
);

impl_filter_gauss_pyramid_layer_up_border_dispatch!(
    f32,
    C1,
    filter_gauss_pyramid_layer_up_border_f32_c1
);

impl_filter_gauss_pyramid_layer_up_border_dispatch!(
    f32,
    C3,
    filter_gauss_pyramid_layer_up_border_f32_c3
);

/// Filters and upsamples a Gaussian-pyramid layer with border control.
///
/// Enqueues work on the selected stream.
/// Ignored channels and row padding may remain unwritten.
///
/// # Errors
///
/// Returns an error if CUDA context binding fails, either image belongs to another CUDA context, or NPP rejects the filter parameters.
/// `source_offset` must address a source pixel, and `rate` must be greater than `1.0` and at most `10.0`.
/// `kernel` must be nonempty and its length must fit in `i32`.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Source samples must be initialized and ready when read by the selected stream.
/// Keep inputs read-only and destination storage exclusively writable and disjoint from all inputs during native use.
/// Retain image allocations through queued completion, including submission errors and captured graph replay.
/// Kernel storage must satisfy the native coefficient address-space requirements and remain readable for as long as native work uses it.
/// Image layouts, sampling footprints, byte strides, alignment, and allocation bounds must satisfy the native routine.
pub unsafe fn filter_gauss_pyramid_layer_up_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    destination: &mut ImageOutputView<'_, T, L>,
    rate: f32,
    kernel: &[f32],
    border_type: BorderType,
) -> Result<()>
where
    T: FilterGaussPyramidLayerUpBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::filter_gauss_pyramid_layer_up_border(
            stream_context,
            source,
            source_offset,
            destination,
            rate,
            kernel,
            border_type,
        )
    }
}
