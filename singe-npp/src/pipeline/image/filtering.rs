pub use crate::pipeline::image::filtering_dispatch::*;
use crate::{
    error::{Error, Result},
    image::{Image, SupportedImage, filtering, view::ChannelLayout},
    pipeline::{ImageBacking, image::ImagePipeline},
    types::{BorderType, MaskSize, Point, PointPolar, Size},
};

mod compressed_marker_labels;
mod outputs;

pub use compressed_marker_labels::{
    CompressedMarkerLabels, ContourGeometry, ContourOrder, Contours, InterpolatedContours,
    MarkerLabels,
};
pub use outputs::{GradientVector, HistogramOfGradients, HoughLines, HoughLinesOutput};

/// Selects the orientation of a one-dimensional image filter.
pub enum FilterAxis {
    /// Filters horizontally along each row.
    Row,
    /// Filters vertically along each column.
    Column,
}

impl FilterAxis {
    pub(crate) fn geometry(self, taps: usize, anchor: i32) -> Result<(Size, Point)> {
        let taps = i32::try_from(taps).map_err(|_| Error::out_of_range("filter taps"))?;
        if !(0..taps).contains(&anchor) {
            return Err(Error::out_of_range("filter anchor"));
        }
        Ok(match self {
            Self::Row => (Size::new(taps, 1), Point { x: anchor, y: 0 }),
            Self::Column => (Size::new(1, taps), Point { x: 0, y: anchor }),
        })
    }
}

/// Returns a square Gaussian mask size and its center anchor.
///
/// # Errors
///
/// Returns an error if `taps` is zero, even, or cannot fit `i32`.
pub fn gaussian_geometry(taps: usize) -> Result<(Size, Point)> {
    let taps = i32::try_from(taps).map_err(|_| Error::out_of_range("gaussian taps"))?;
    if taps <= 0 || taps % 2 == 0 {
        return Err(Error::out_of_range("gaussian taps"));
    }
    Ok((
        Size::new(taps, taps),
        Point {
            x: taps / 2,
            y: taps / 2,
        },
    ))
}

/// Returns the square mask size and center anchor for `mask`.
///
/// # Errors
///
/// Returns an error if `mask` is a one-dimensional mask rather than a square mask.
pub fn centered_filter_geometry(mask: MaskSize) -> Result<(Size, Point)> {
    let taps = match mask {
        MaskSize::Mask3x3 => 3,
        MaskSize::Mask5x5 => 5,
        MaskSize::Mask7x7 => 7,
        MaskSize::Mask9x9 => 9,
        MaskSize::Mask11x11 => 11,
        MaskSize::Mask13x13 => 13,
        MaskSize::Mask15x15 => 15,
        MaskSize::Mask1x3 | MaskSize::Mask1x5 | MaskSize::Mask3x1 | MaskSize::Mask5x1 => {
            return Err(Error::out_of_range("square filter mask"));
        }
    };
    gaussian_geometry(taps)
}

impl<T, L> ImagePipeline<'_, T, L> {
    /// Returns the required scratch capacity in bytes for union-find labeling of `roi`.
    ///
    /// The provided query forwards the ROI dimensions to NPP.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the query or the returned byte count cannot be represented as `usize`.
    ///
    /// See [`label_markers_uf_buffer_size_u32_c1`](crate::image::raw::filtering::label_markers_uf_buffer_size_u32_c1) for the underlying query.
    pub fn label_markers_uf_buffer_size(roi: Size) -> Result<usize> {
        filtering::label_markers_uf_buffer_size_u32_c1(roi)
    }

    /// Returns the scratch capacity in bytes for compressing UF marker labels.
    ///
    /// `starting_number` is the ROI width times height used during marker generation.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the query or the returned byte count cannot be represented as `usize`.
    ///
    /// See [`compress_marker_labels_uf_buffer_size_u32_c1`](crate::image::raw::filtering::compress_marker_labels_uf_buffer_size_u32_c1) for the underlying query.
    pub fn compress_marker_labels_uf_buffer_size(starting_number: i32) -> Result<usize> {
        filtering::compress_marker_labels_uf_buffer_size_u32_c1(starting_number)
    }

    /// Returns the device scratch size in bytes for flood filling an image of size `roi`.
    ///
    /// # Errors
    ///
    /// Returns an error if either ROI dimension is nonpositive, NPP rejects the size query, or the returned size cannot be represented as `usize`.
    ///
    /// See [`flood_fill_buffer_size`](crate::image::raw::filtering::flood_fill_buffer_size) for the underlying query.
    pub fn flood_fill_buffer_size(roi: Size) -> Result<usize> {
        filtering::flood_fill_buffer_size(roi)
    }

    /// Returns the scratch capacity in bytes for Canny edge detection.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the ROI or query parameters, or the returned size cannot be represented as `usize`.
    ///
    /// See [`filter_canny_border_buffer_size`](crate::image::raw::filtering::filter_canny_border_buffer_size) for the underlying query.
    pub fn filter_canny_border_buffer_size(roi: Size) -> Result<usize> {
        filtering::filter_canny_border_buffer_size(roi)
    }

    /// Returns the scratch capacity in bytes for Harris corner responses.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the ROI or query parameters, or the returned size cannot be represented as `usize`.
    ///
    /// See [`filter_harris_corners_border_buffer_size`](crate::image::raw::filtering::filter_harris_corners_border_buffer_size) for the underlying query.
    pub fn filter_harris_corners_border_buffer_size(roi: Size) -> Result<usize> {
        filtering::filter_harris_corners_border_buffer_size(roi)
    }

    /// Returns the scratch capacity in bytes for Hough line detection.
    /// `delta` supplies the native radial and angular discretization increments; `max_line_count` is the output capacity.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the ROI or query parameters, or the returned size cannot be represented as `usize`.
    /// ROI dimensions and `max_line_count` must be positive. `delta.rho` must lie strictly between zero and three, and `delta.theta` strictly between 0.25 and three.
    ///
    /// See [`filter_hough_line_buffer_size`](crate::image::raw::filtering::filter_hough_line_buffer_size) for the underlying query.
    pub fn filter_hough_line_buffer_size(
        roi: Size,
        delta: PointPolar,
        max_line_count: i32,
    ) -> Result<usize> {
        filtering::filter_hough_line_buffer_size(roi, delta, max_line_count)
    }

    /// Returns the destination ROI for Gaussian-pyramid downsampling.
    ///
    /// # Errors
    ///
    /// Returns an error if a source dimension is nonpositive, `rate` is not greater than 1 and at most 10, or NPP rejects the query.
    pub fn filter_gauss_pyramid_layer_down_border_destination_roi(
        source_roi: Size,
        rate: f32,
    ) -> Result<Size> {
        filtering::filter_gauss_pyramid_layer_down_border_dst_roi(source_roi, rate)
    }

    /// Returns the minimum and maximum destination ROIs for Gaussian-pyramid upsampling.
    ///
    /// # Errors
    ///
    /// Returns an error if a source dimension is nonpositive, `rate` is not greater than 1 and at most 10, or NPP rejects the query.
    pub fn filter_gauss_pyramid_layer_up_border_destination_roi(
        source_roi: Size,
        rate: f32,
    ) -> Result<(Size, Size)> {
        filtering::filter_gauss_pyramid_layer_up_border_dst_roi(source_roi, rate)
    }

    /// Returns scratch size in bytes for PBA distance transformation on `roi`.
    ///
    /// # Errors
    ///
    /// Returns an error if either ROI dimension is nonpositive, NPP rejects the size query, or the returned size cannot be represented as `usize`.
    ///
    /// See [`distance_transform_pba_buffer_size`](crate::image::raw::filtering::distance_transform_pba_buffer_size) for the underlying query.
    pub fn distance_transform_pba_buffer_size(roi: Size) -> Result<usize> {
        filtering::distance_transform_pba_buffer_size(roi)
    }

    /// Returns antialiasing scratch size in bytes for PBA distance transformation on `roi`.
    ///
    /// # Errors
    ///
    /// Returns an error if either ROI dimension is nonpositive, NPP rejects the size query, or the returned size cannot be represented as `usize`.
    ///
    /// See [`distance_transform_pba_antialiasing_buffer_size`](crate::image::raw::filtering::distance_transform_pba_antialiasing_buffer_size) for the underlying query.
    pub fn distance_transform_pba_antialiasing_buffer_size(roi: Size) -> Result<usize> {
        filtering::distance_transform_pba_antialiasing_buffer_size(roi)
    }

    /// Returns scratch size in bytes for signed PBA distance transformation on `roi`.
    ///
    /// # Errors
    ///
    /// Returns an error if either ROI dimension is nonpositive, NPP rejects the size query, or the returned size cannot be represented as `usize`.
    ///
    /// See [`signed_distance_transform_pba_buffer_size`](crate::image::raw::filtering::signed_distance_transform_pba_buffer_size) for the underlying query.
    pub fn signed_distance_transform_pba_buffer_size(roi: Size) -> Result<usize> {
        filtering::signed_distance_transform_pba_buffer_size(roi)
    }

    /// Returns scratch size in bytes for signed double-precision PBA distance transformation on `roi`.
    ///
    /// # Errors
    ///
    /// Returns an error if either ROI dimension is nonpositive, NPP rejects the size query, or the returned size cannot be represented as `usize`.
    ///
    /// See [`signed_distance_transform_pba_64f_buffer_size`](crate::image::raw::filtering::signed_distance_transform_pba_64f_buffer_size) for the underlying query.
    pub fn signed_distance_transform_pba_64f_buffer_size(roi: Size) -> Result<usize> {
        filtering::signed_distance_transform_pba_64f_buffer_size(roi)
    }

    /// Returns antialiasing scratch size in bytes for signed PBA distance transformation on `roi`.
    ///
    /// # Errors
    ///
    /// Returns an error if either ROI dimension is nonpositive, NPP rejects the size query, or the returned size cannot be represented as `usize`.
    ///
    /// See [`signed_distance_transform_pba_antialiasing_buffer_size`](crate::image::raw::filtering::signed_distance_transform_pba_antialiasing_buffer_size) for the underlying query.
    pub fn signed_distance_transform_pba_antialiasing_buffer_size(roi: Size) -> Result<usize> {
        filtering::signed_distance_transform_pba_antialiasing_buffer_size(roi)
    }

    /// Returns the scratch capacity in bytes for watershed segmentation of `roi`.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the ROI or query parameters, or the returned byte count cannot be represented as `usize`.
    ///
    /// See [`segment_watershed_buffer_size_u8_c1`](crate::image::raw::filtering::segment_watershed_buffer_size_u8_c1) for the underlying query.
    pub fn segment_watershed_u8_buffer_size(roi: Size) -> Result<usize> {
        filtering::segment_watershed_buffer_size_u8_c1(roi)
    }

    /// Returns the scratch capacity in bytes for watershed segmentation of `roi`.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the ROI or query parameters, or the returned byte count cannot be represented as `usize`.
    ///
    /// See [`segment_watershed_buffer_size_u16_c1`](crate::image::raw::filtering::segment_watershed_buffer_size_u16_c1) for the underlying query.
    pub fn segment_watershed_u16_buffer_size(roi: Size) -> Result<usize> {
        filtering::segment_watershed_buffer_size_u16_c1(roi)
    }

    /// Returns the scratch capacity in bytes for watershed segmentation of `roi`.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the ROI or query parameters, or the returned byte count cannot be represented as `usize`.
    ///
    /// See [`segment_watershed_buffer_size_for`](crate::image::raw::filtering::segment_watershed_buffer_size_for) for the underlying query.
    pub fn segment_watershed_buffer_size<U: filtering::SegmentWatershedC1InPlace>(
        roi: Size,
    ) -> Result<usize> {
        filtering::segment_watershed_buffer_size_for::<U>(roi)
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: AdaptiveBoxThresholdBorderImage<T, L>,
{
    /// Thresholds each channel against its local box mean minus `delta` into a new owned image.
    ///
    /// Pixels above the threshold become `value_greater_than`; other pixels become
    /// `value_less_or_equal`. The output covers the source window from `source_offset` to the
    /// image's lower-right corner. NPP requires a square mask with odd side lengths and supports
    /// only [`BorderType::Replicate`]. For layouts that ignore
    /// alpha, the source window is copied first to preserve its alpha values.
    /// The operation prepares source and destination memory and completes queued work before
    /// returning the resulting pipeline.
    ///
    /// # Errors
    ///
    /// Returns an error during stream capture, if the source view or offset is invalid,
    /// allocation or output-view creation fails, preserving alpha fails, region preparation
    /// fails, or the native threshold operation or stream synchronization fails. Native mask
    /// and border restrictions are those of the selected adaptive-box threshold operation.
    pub fn filter_threshold_adaptive_box_border(
        self,
        source_offset: Point,
        mask_size: Size,
        delta: f32,
        value_greater_than: T,
        value_less_or_equal: T,
        border_type: BorderType,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_source = source.window_from(source_offset)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), output_source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&output_source.view())?;
            }
            <Self as AdaptiveBoxThresholdBorderImage<T, L>>::filter_threshold_adaptive_box_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask_size,
                delta,
                value_greater_than,
                value_less_or_equal,
                border_type,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
