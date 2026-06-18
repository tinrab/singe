use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, C3, ImageView, ImageViewMut},
    },
    types::{BorderType, DifferentialKernel, ImageNormalization, MaskSize, Point},
};

use super::ImagePipeline;

use super::filtering_traits::*;

macro_rules! impl_canny_border_filter_image {
    ($ty:ty, $layout:ty, $filter_canny:path) => {
        impl<'a> CannyBorderFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_canny_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, u8, C1>,
                filter_type: DifferentialKernel,
                mask_size: MaskSize,
                low_threshold: i16,
                high_threshold: i16,
                norm: ImageNormalization,
                border_type: BorderType,
            ) -> Result<()> {
                $filter_canny(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    filter_type,
                    mask_size,
                    low_threshold,
                    high_threshold,
                    norm,
                    border_type,
                )
            }
        }
    };
}

macro_rules! impl_harris_corners_border_filter_image {
    ($ty:ty, $layout:ty, $filter_harris:path) => {
        impl<'a> HarrisCornersBorderFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_harris_corners_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, f32, C1>,
                filter_type: DifferentialKernel,
                mask_size: MaskSize,
                average_window_size: MaskSize,
                k: f32,
                scale: f32,
                border_type: BorderType,
            ) -> Result<()> {
                $filter_harris(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    filter_type,
                    mask_size,
                    average_window_size,
                    k,
                    scale,
                    border_type,
                )
            }
        }
    };
}

impl_canny_border_filter_image!(u8, C1, filtering::filter_canny_border);
impl_canny_border_filter_image!(u8, C3, filtering::filter_canny_border_to_mask);

impl_harris_corners_border_filter_image!(u8, C1, filtering::filter_harris_corners_border_to_f32);

#[path = "filtering_dispatch_gradient_vector.rs"]
mod gradient_vector;
#[path = "filtering_dispatch_histogram_of_gradients.rs"]
mod histogram_of_gradients;
