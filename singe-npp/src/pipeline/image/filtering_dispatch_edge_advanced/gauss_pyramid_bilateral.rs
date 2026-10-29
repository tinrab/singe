use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, C3, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImagePipeline,
        filtering_traits::{BilateralGaussBorderFilterImage, GaussPyramidBorderFilterImage},
    },
    types::{BorderType, Point},
};

macro_rules! impl_gauss_pyramid_border_filter_image {
    ($ty:ty, $layout:ty, $down:path, $up:path) => {
        unsafe impl<'a> GaussPyramidBorderFilterImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn filter_gauss_pyramid_layer_down_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                rate: f32,
                kernel: &[f32],
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $down(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        rate,
                        kernel,
                        border_type,
                    )
                })
            }

            fn filter_gauss_pyramid_layer_up_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                rate: f32,
                kernel: &[f32],
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $up(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        rate,
                        kernel,
                        border_type,
                    )
                })
            }
        }
    };
}

macro_rules! impl_bilateral_gauss_border_filter_image {
    ($ty:ty, $layout:ty, $bilateral:path) => {
        unsafe impl<'a> BilateralGaussBorderFilterImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn filter_bilateral_gauss_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                radius: i32,
                step_between_source_pixels: i32,
                value_square_sigma: f32,
                position_square_sigma: f32,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $bilateral(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        radius,
                        step_between_source_pixels,
                        value_square_sigma,
                        position_square_sigma,
                        border_type,
                    )
                })
            }
        }
    };
}

impl_gauss_pyramid_border_filter_image!(
    u8,
    C1,
    filtering::filter_gauss_pyramid_layer_down_border_u8_c1,
    filtering::filter_gauss_pyramid_layer_up_border_u8_c1
);

impl_gauss_pyramid_border_filter_image!(
    u8,
    C3,
    filtering::filter_gauss_pyramid_layer_down_border_u8_c3,
    filtering::filter_gauss_pyramid_layer_up_border_u8_c3
);

impl_gauss_pyramid_border_filter_image!(
    u16,
    C1,
    filtering::filter_gauss_pyramid_layer_down_border_u16_c1,
    filtering::filter_gauss_pyramid_layer_up_border_u16_c1
);

impl_gauss_pyramid_border_filter_image!(
    u16,
    C3,
    filtering::filter_gauss_pyramid_layer_down_border_u16_c3,
    filtering::filter_gauss_pyramid_layer_up_border_u16_c3
);

impl_gauss_pyramid_border_filter_image!(
    f32,
    C1,
    filtering::filter_gauss_pyramid_layer_down_border_f32_c1,
    filtering::filter_gauss_pyramid_layer_up_border_f32_c1
);

impl_gauss_pyramid_border_filter_image!(
    f32,
    C3,
    filtering::filter_gauss_pyramid_layer_down_border_f32_c3,
    filtering::filter_gauss_pyramid_layer_up_border_f32_c3
);

impl_bilateral_gauss_border_filter_image!(u8, C1, filtering::filter_bilateral_gauss_border_u8_c1);

impl_bilateral_gauss_border_filter_image!(u8, C3, filtering::filter_bilateral_gauss_border_u8_c3);

impl_bilateral_gauss_border_filter_image!(u16, C1, filtering::filter_bilateral_gauss_border_u16_c1);

impl_bilateral_gauss_border_filter_image!(u16, C3, filtering::filter_bilateral_gauss_border_u16_c3);

impl_bilateral_gauss_border_filter_image!(f32, C1, filtering::filter_bilateral_gauss_border_f32_c1);

impl_bilateral_gauss_border_filter_image!(f32, C3, filtering::filter_bilateral_gauss_border_f32_c3);
