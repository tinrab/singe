use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView},
    },
    pipeline::image::{ImagePipeline, filtering_traits::DirectionalBorderFilterImage},
    types::{BorderType, Point},
};

macro_rules! impl_directional_border_filter_image {
    ($ty:ty, $layout:ty, $sharpen:path) => {
        unsafe impl<'a> DirectionalBorderFilterImage<$ty, $layout>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn filter_sharpen_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $sharpen(
                        stream_context,
                        source,
                        source_offset,
                        destination,
                        border_type,
                    )
                })
            }
        }
    };
}

impl_directional_border_filter_image!(u8, C1, filtering::filter_sharpen_border_u8_c1);

impl_directional_border_filter_image!(u8, C3, filtering::filter_sharpen_border_u8_c3);

impl_directional_border_filter_image!(u8, C4, filtering::filter_sharpen_border_u8_c4);

impl_directional_border_filter_image!(u8, AC4, filtering::filter_sharpen_border_u8_ac4);

impl_directional_border_filter_image!(u16, C1, filtering::filter_sharpen_border_u16_c1);

impl_directional_border_filter_image!(u16, C3, filtering::filter_sharpen_border_u16_c3);

impl_directional_border_filter_image!(u16, C4, filtering::filter_sharpen_border_u16_c4);

impl_directional_border_filter_image!(u16, AC4, filtering::filter_sharpen_border_u16_ac4);

impl_directional_border_filter_image!(i16, C1, filtering::filter_sharpen_border_i16_c1);

impl_directional_border_filter_image!(i16, C3, filtering::filter_sharpen_border_i16_c3);

impl_directional_border_filter_image!(i16, C4, filtering::filter_sharpen_border_i16_c4);

impl_directional_border_filter_image!(i16, AC4, filtering::filter_sharpen_border_i16_ac4);

impl_directional_border_filter_image!(f32, C1, filtering::filter_sharpen_border_f32_c1);

impl_directional_border_filter_image!(f32, C3, filtering::filter_sharpen_border_f32_c3);

impl_directional_border_filter_image!(f32, C4, filtering::filter_sharpen_border_f32_c4);

impl_directional_border_filter_image!(f32, AC4, filtering::filter_sharpen_border_f32_ac4);
