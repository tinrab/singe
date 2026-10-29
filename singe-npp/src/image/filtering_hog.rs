use singe_cuda::view::{DeviceOutputView, DeviceRegionProvider};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        filtering_validation::{validate_border_roi, validate_positive_size},
        view::{C1, C3, ChannelLayout, ImageView},
    },
    try_ffi,
    types::{BorderType, DataTypeLike, HistogramOfGradientsConfig, Point, Size},
    utility::to_usize,
};

/// Returns the device scratch capacity in bytes for HOG descriptor extraction.
///
/// Use the same `config`, host `locations` slice, and source `roi` as the extraction call.
/// NPP validates the configuration and requested windows; this query does not allocate memory.
///
/// # Errors
///
/// Returns an error if either ROI dimension is nonpositive, the location count does not fit
/// `i32`, NPP rejects the request, or the returned byte count cannot be represented as `usize`.
pub fn histogram_of_gradients_border_buffer_size(
    config: HistogramOfGradientsConfig,
    locations: &[Point],
    roi: Size,
) -> Result<usize> {
    validate_positive_size(roi)?;
    let location_count =
        i32::try_from(locations.len()).map_err(|_| Error::out_of_range("location count"))?;
    let mut bytes = 0;
    // SAFETY: `nppiHistogramOfGradientsBorderGetBufferSize` receives valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiHistogramOfGradientsBorderGetBufferSize(
            config.into(),
            locations.as_ptr().cast(),
            location_count,
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

/// Returns the total device output capacity in bytes for HOG descriptors.
///
/// NPP validates `config` and `location_count`. Each location produces one descriptor window;
/// the windows are stored consecutively without padding as `f32` values. The returned size
/// counts bytes, not `f32` elements. This query does not allocate memory.
///
/// # Errors
///
/// Returns an error if `location_count` does not fit `i32`, NPP rejects the request,
/// or the returned byte count cannot be represented as `usize`.
pub fn histogram_of_gradients_border_descriptors_size(
    config: HistogramOfGradientsConfig,
    location_count: usize,
) -> Result<usize> {
    let location_count =
        i32::try_from(location_count).map_err(|_| Error::out_of_range("location count"))?;
    let mut bytes = 0;
    // SAFETY: `nppiHistogramOfGradientsBorderGetDescriptorsSize` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiHistogramOfGradientsBorderGetDescriptorsSize(
            config.into(),
            location_count,
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "descriptors size")
}

macro_rules! impl_histogram_of_gradients_border {
    ($name:ident, $pixel_ty:ty, $layout:ty, $ffi_name:ident) => {
        /// Enqueues histogram-of-oriented-gradients descriptors for the requested source windows.
        ///
        /// `locations` is a host slice of source pixel starting locations. `source_offset` identifies
        /// the pixel within the full source image at which the ROI begins; the wrapper offsets the
        /// native source pointer accordingly. Only [`BorderType::Replicate`] is supported by NPP.
        ///
        /// The operation uses centered derivatives, unsigned orientations from 0 to 180 degrees,
        /// linear interpolation between the two nearest bins, and overlapping blocks with L2
        /// normalization. It performs no threshold clipping or Gaussian smoothing. For three-channel
        /// images, the channel with the greatest gradient magnitude supplies each pixel's gradient.
        /// Descriptor windows contain `f32` bins in row order and are packed consecutively without padding.
        ///
        /// `descriptors` must have at least the byte capacity returned by
        /// [`histogram_of_gradients_border_descriptors_size`]. `scratch` must have at least the
        /// capacity returned by [`histogram_of_gradients_border_buffer_size`] for the same request.
        /// NPP validates configuration and window limits through these queries. This call checks
        /// capacities and CUDA contexts, then queues work without synchronizing the stream.
        ///
        /// # Errors
        ///
        /// Returns an error if the source offset or ROI is invalid, a size query fails, either output
        /// capacity is too small, a location count cannot fit `i32`, binding the CUDA context fails,
        /// a source or output region belongs to a different CUDA context, or NPP rejects the operation.
        ///
        /// # Safety
        ///
        /// The source pixels must be initialized and ready for reading on `stream_context`.
        /// Keep the source, descriptors, and scratch allocations valid until queued work completes,
        /// including every replay if captured. Prevent conflicting access from other streams or the
        /// host during that time; source, descriptor output, and scratch storage must not overlap.
        /// The descriptor and scratch regions must permit exclusive device writes. The host
        /// `locations` slice must remain readable for this call. Uphold the native configuration and
        /// window requirements not established by the size queries.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            source: &ImageView<'_, $pixel_ty, $layout>,
            source_offset: Point,
            locations: &[Point],
            descriptors: &mut DeviceOutputView<'_, f32>,
            roi: Size,
            config: HistogramOfGradientsConfig,
            scratch: &mut DeviceOutputView<'_, u8>,
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
            stream_context.validate_region(descriptors)?;
            stream_context.validate_region(scratch)?;

            validate_border_roi(source.size(), source_offset, roi)?;
            let descriptor_bytes =
                histogram_of_gradients_border_descriptors_size(config, locations.len())?;
            if descriptors.region().byte_len() < descriptor_bytes {
                return Err(Error::out_of_range("descriptors length"));
            }
            let scratch_bytes = histogram_of_gradients_border_buffer_size(config, locations, roi)?;
            if scratch.len() < scratch_bytes {
                return Err(Error::out_of_range("scratch length"));
            }
            let location_count = i32::try_from(locations.len())
                .map_err(|_| Error::out_of_range("location count"))?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    source_pointer.cast(),
                    source.step(),
                    source.size().into(),
                    source_offset.into(),
                    locations.as_ptr().cast(),
                    location_count,
                    descriptors.as_mut_ptr().cast(),
                    roi.into(),
                    config.into(),
                    scratch.as_mut_ptr().cast(),
                    border_type.into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_u8_to_f32_c1,
    u8,
    C1,
    nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx
);

impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_u8_to_f32_c3,
    u8,
    C3,
    nppiHistogramOfGradientsBorder_8u32f_C3R_Ctx
);

impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_u16_to_f32_c1,
    u16,
    C1,
    nppiHistogramOfGradientsBorder_16u32f_C1R_Ctx
);

impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_u16_to_f32_c3,
    u16,
    C3,
    nppiHistogramOfGradientsBorder_16u32f_C3R_Ctx
);

impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_i16_to_f32_c1,
    i16,
    C1,
    nppiHistogramOfGradientsBorder_16s32f_C1R_Ctx
);

impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_i16_to_f32_c3,
    i16,
    C3,
    nppiHistogramOfGradientsBorder_16s32f_C3R_Ctx
);

impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_f32_c1,
    f32,
    C1,
    nppiHistogramOfGradientsBorder_32f_C1R_Ctx
);

impl_histogram_of_gradients_border!(
    histogram_of_gradients_border_f32_c3,
    f32,
    C3,
    nppiHistogramOfGradientsBorder_32f_C3R_Ctx
);

/// Dispatches HOG descriptor extraction for a native pixel type and channel layout.
///
/// # Safety
///
/// Implementations must preserve the pixel ABI and channel layout expected by the selected
/// native operation and uphold the method's validation and queued execution contract.
pub unsafe trait HistogramOfGradientsBorder<L: ChannelLayout>: DataTypeLike {
    /// Enqueues histogram-of-oriented-gradients descriptors for the requested source windows.
    ///
    /// `locations` is a host slice of source pixel starting locations. `source_offset` identifies
    /// the pixel within the full source image at which the ROI begins; the wrapper offsets the
    /// native source pointer accordingly. Only [`BorderType::Replicate`] is supported by NPP.
    ///
    /// The operation uses centered derivatives, unsigned orientations from 0 to 180 degrees,
    /// linear interpolation between the two nearest bins, and overlapping blocks with L2
    /// normalization. It performs no threshold clipping or Gaussian smoothing. For three-channel
    /// images, the channel with the greatest gradient magnitude supplies each pixel's gradient.
    /// Descriptor windows contain `f32` bins in row order and are packed consecutively without padding.
    ///
    /// `descriptors` must have at least the byte capacity returned by
    /// [`histogram_of_gradients_border_descriptors_size`]. `scratch` must have at least the
    /// capacity returned by [`histogram_of_gradients_border_buffer_size`] for the same request.
    /// NPP validates configuration and window limits through these queries. This call checks
    /// capacities and CUDA contexts, then queues work without synchronizing the stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the source offset or ROI is invalid, a size query fails, either output
    /// capacity is too small, a location count cannot fit `i32`, binding the CUDA context fails,
    /// a source or output region belongs to a different CUDA context, or NPP rejects the operation.
    ///
    /// # Safety
    ///
    /// The source pixels must be initialized and ready for reading on `stream_context`.
    /// Keep the source, descriptors, and scratch allocations valid until queued work completes,
    /// including every replay if captured. Prevent conflicting access from other streams or the
    /// host during that time; source, descriptor output, and scratch storage must not overlap.
    /// The descriptor and scratch regions must permit exclusive device writes. The host
    /// `locations` slice must remain readable for this call. Uphold the native configuration and
    /// window requirements not established by the size queries.
    unsafe fn histogram_of_gradients_border(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        source_offset: Point,
        locations: &[Point],
        descriptors: &mut DeviceOutputView<'_, f32>,
        roi: Size,
        config: HistogramOfGradientsConfig,
        scratch: &mut DeviceOutputView<'_, u8>,
        border_type: BorderType,
    ) -> Result<()>;
}

macro_rules! impl_histogram_of_gradients_border_dispatch {
    ($pixel_ty:ty, $layout:ty, $name:ident) => {
        unsafe impl HistogramOfGradientsBorder<$layout> for $pixel_ty {
            unsafe fn histogram_of_gradients_border(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                source_offset: Point,
                locations: &[Point],
                descriptors: &mut DeviceOutputView<'_, f32>,
                roi: Size,
                config: HistogramOfGradientsConfig,
                scratch: &mut DeviceOutputView<'_, u8>,
                border_type: BorderType,
            ) -> Result<()> {
                // SAFETY: the caller upholds this function's safety contract.
                unsafe {
                    $name(
                        stream_context,
                        source,
                        source_offset,
                        locations,
                        descriptors,
                        roi,
                        config,
                        scratch,
                        border_type,
                    )
                }
            }
        }
    };
}

impl_histogram_of_gradients_border_dispatch!(u8, C1, histogram_of_gradients_border_u8_to_f32_c1);

impl_histogram_of_gradients_border_dispatch!(u8, C3, histogram_of_gradients_border_u8_to_f32_c3);

impl_histogram_of_gradients_border_dispatch!(u16, C1, histogram_of_gradients_border_u16_to_f32_c1);

impl_histogram_of_gradients_border_dispatch!(u16, C3, histogram_of_gradients_border_u16_to_f32_c3);

impl_histogram_of_gradients_border_dispatch!(i16, C1, histogram_of_gradients_border_i16_to_f32_c1);

impl_histogram_of_gradients_border_dispatch!(i16, C3, histogram_of_gradients_border_i16_to_f32_c3);

impl_histogram_of_gradients_border_dispatch!(f32, C1, histogram_of_gradients_border_f32_c1);

impl_histogram_of_gradients_border_dispatch!(f32, C3, histogram_of_gradients_border_f32_c3);

/// Enqueues histogram-of-oriented-gradients descriptors for the requested source windows.
///
/// `locations` is a host slice of source pixel starting locations. `source_offset` identifies
/// the pixel within the full source image at which the ROI begins; the wrapper offsets the
/// native source pointer accordingly. Only [`BorderType::Replicate`] is supported by NPP.
///
/// The operation uses centered derivatives, unsigned orientations from 0 to 180 degrees,
/// linear interpolation between the two nearest bins, and overlapping blocks with L2
/// normalization. It performs no threshold clipping or Gaussian smoothing. For three-channel
/// images, the channel with the greatest gradient magnitude supplies each pixel's gradient.
/// Descriptor windows contain `f32` bins in row order and are packed consecutively without padding.
///
/// `descriptors` must have at least the byte capacity returned by
/// [`histogram_of_gradients_border_descriptors_size`]. `scratch` must have at least the
/// capacity returned by [`histogram_of_gradients_border_buffer_size`] for the same request.
/// NPP validates configuration and window limits through these queries. This call checks
/// capacities and CUDA contexts, then queues work without synchronizing the stream.
///
/// # Errors
///
/// Returns an error if the source offset or ROI is invalid, a size query fails, either output
/// capacity is too small, a location count cannot fit `i32`, binding the CUDA context fails,
/// a source or output region belongs to a different CUDA context, or NPP rejects the operation.
///
/// # Safety
///
/// The source pixels must be initialized and ready for reading on `stream_context`.
/// Keep the source, descriptors, and scratch allocations valid until queued work completes,
/// including every replay if captured. Prevent conflicting access from other streams or the
/// host during that time; source, descriptor output, and scratch storage must not overlap.
/// The descriptor and scratch regions must permit exclusive device writes. The host
/// `locations` slice must remain readable for this call. Uphold the native configuration and
/// window requirements not established by the size queries.
pub unsafe fn histogram_of_gradients_border<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    source_offset: Point,
    locations: &[Point],
    descriptors: &mut DeviceOutputView<'_, f32>,
    roi: Size,
    config: HistogramOfGradientsConfig,
    scratch: &mut DeviceOutputView<'_, u8>,
    border_type: BorderType,
) -> Result<()>
where
    T: HistogramOfGradientsBorder<L>,
    L: ChannelLayout,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        T::histogram_of_gradients_border(
            stream_context,
            source,
            source_offset,
            locations,
            descriptors,
            roi,
            config,
            scratch,
            border_type,
        )
    }
}
