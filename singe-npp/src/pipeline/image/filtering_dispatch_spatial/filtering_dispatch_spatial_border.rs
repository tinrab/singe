use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImagePipeline,
        filtering_traits::{
            AdaptiveBoxThresholdBorderImage, BoxBorderFilterImage, MedianBorderFilterImage,
        },
    },
    types::{BorderType, Point, Size},
};

macro_rules! impl_box_border_filter_image {
    ($ty:ty, $layout:ty, $filter_box:path) => {
        unsafe impl<'a> BoxBorderFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_box_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $filter_box(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask_size,
                        anchor,
                        border_type,
                    )
                })
            }
        }
    };
}

macro_rules! impl_adaptive_box_threshold_border_image {
    ($ty:ty, $layout:ty, $filter:path) => {
        unsafe impl<'a> AdaptiveBoxThresholdBorderImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn filter_threshold_adaptive_box_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask_size: Size,
                delta: f32,
                value_greater_than: $ty,
                value_less_or_equal: $ty,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $filter(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask_size,
                        delta,
                        value_greater_than,
                        value_less_or_equal,
                        border_type,
                    )
                })
            }
        }
    };
}

mod neighborhood_border;

macro_rules! impl_median_border_filter_image {
    ($ty:ty, $layout:ty, $filter_median:path) => {
        unsafe impl<'a> MedianBorderFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_median_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask_size: Size,
                anchor: Point,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the native helper completes borrowed use on every path.
                unsafe {
                    $filter_median(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        mask_size,
                        anchor,
                        border_type,
                    )
                }
            }
        }
    };
}

impl_box_border_filter_image!(u8, C1, filtering::filter_box_border_u8_c1);

impl_box_border_filter_image!(u8, C3, filtering::filter_box_border_u8_c3);

impl_box_border_filter_image!(u8, C4, filtering::filter_box_border_u8_c4);

impl_box_border_filter_image!(u8, AC4, filtering::filter_box_border_u8_ac4);

impl_box_border_filter_image!(u16, C1, filtering::filter_box_border_u16_c1);

impl_box_border_filter_image!(u16, C3, filtering::filter_box_border_u16_c3);

impl_box_border_filter_image!(u16, C4, filtering::filter_box_border_u16_c4);

impl_box_border_filter_image!(u16, AC4, filtering::filter_box_border_u16_ac4);

impl_box_border_filter_image!(i16, C1, filtering::filter_box_border_i16_c1);

impl_box_border_filter_image!(i16, C3, filtering::filter_box_border_i16_c3);

impl_box_border_filter_image!(i16, C4, filtering::filter_box_border_i16_c4);

impl_box_border_filter_image!(i16, AC4, filtering::filter_box_border_i16_ac4);

impl_box_border_filter_image!(f32, C1, filtering::filter_box_border_f32_c1);

impl_box_border_filter_image!(f32, C3, filtering::filter_box_border_f32_c3);

impl_box_border_filter_image!(f32, C4, filtering::filter_box_border_f32_c4);

impl_box_border_filter_image!(f32, AC4, filtering::filter_box_border_f32_ac4);

impl_adaptive_box_threshold_border_image!(u8, C1, filtering::filter_threshold_adaptive_box_border);

impl_median_border_filter_image!(u8, C1, filtering::filter_median_border_u8_c1);

impl_median_border_filter_image!(u8, C3, filtering::filter_median_border_u8_c3);

impl_median_border_filter_image!(u8, C4, filtering::filter_median_border_u8_c4);

impl_median_border_filter_image!(u8, AC4, filtering::filter_median_border_u8_ac4);

impl_median_border_filter_image!(u16, C1, filtering::filter_median_border_u16_c1);

impl_median_border_filter_image!(u16, C3, filtering::filter_median_border_u16_c3);

impl_median_border_filter_image!(u16, C4, filtering::filter_median_border_u16_c4);

impl_median_border_filter_image!(u16, AC4, filtering::filter_median_border_u16_ac4);

impl_median_border_filter_image!(i16, C1, filtering::filter_median_border_i16_c1);

impl_median_border_filter_image!(i16, C4, filtering::filter_median_border_i16_c4);

impl_median_border_filter_image!(i16, AC4, filtering::filter_median_border_i16_ac4);

impl_median_border_filter_image!(f32, C1, filtering::filter_median_border_f32_c1);

impl_median_border_filter_image!(f32, C3, filtering::filter_median_border_f32_c3);

impl_median_border_filter_image!(f32, C4, filtering::filter_median_border_f32_c4);

impl_median_border_filter_image!(f32, AC4, filtering::filter_median_border_f32_ac4);
