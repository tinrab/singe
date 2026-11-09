use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, C3, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImagePipeline,
        filtering_traits::{CannyBorderFilterImage, HarrisCornersBorderFilterImage},
    },
    types::{BorderType, DifferentialKernel, ImageNormalization, MaskSize, Point},
};

macro_rules! impl_canny_border_filter_image {
    ($ty:ty, $layout:ty, $filter_canny:path) => {
        unsafe impl<'a> CannyBorderFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_canny_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, u8, C1>,
                filter_type: DifferentialKernel,
                mask_size: MaskSize,
                low_threshold: i16,
                high_threshold: i16,
                norm: ImageNormalization,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
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
                })
            }
        }
    };
}

macro_rules! impl_harris_corners_border_filter_image {
    ($ty:ty, $layout:ty, $filter_harris:path) => {
        unsafe impl<'a> HarrisCornersBorderFilterImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn filter_harris_corners_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, f32, C1>,
                filter_type: DifferentialKernel,
                mask_size: MaskSize,
                average_window_size: MaskSize,
                k: f32,
                scale: f32,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
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
                })
            }
        }
    };
}

impl_canny_border_filter_image!(u8, C1, filtering::filter_canny_border);

impl_canny_border_filter_image!(u8, C3, filtering::filter_canny_border_to_mask);

impl_harris_corners_border_filter_image!(u8, C1, filtering::filter_harris_corners_border_to_f32);

mod gradient_vector;
mod histogram_of_gradients;
