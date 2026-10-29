use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageWindow},
    },
    pipeline::image::{ImagePipeline, filtering_traits::DirectionalFilterImage},
};

macro_rules! impl_directional_filter_image {
    ($ty:ty, $layout:ty, $sharpen:path) => {
        unsafe impl<'a> DirectionalFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_sharpen_image(
                stream_context: &StreamContext,
                source: &ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                source.require_halo(
                    $crate::types::Size::new(3, 3),
                    $crate::types::Point { x: 1, y: 1 },
                )?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $sharpen(stream_context, &source.view(), destination) })
            }
        }
    };
}

impl_directional_filter_image!(u8, C1, filtering::filter_sharpen_u8_c1);

impl_directional_filter_image!(u8, C3, filtering::filter_sharpen_u8_c3);

impl_directional_filter_image!(u8, C4, filtering::filter_sharpen_u8_c4);

impl_directional_filter_image!(u8, AC4, filtering::filter_sharpen_u8_ac4);

impl_directional_filter_image!(u16, C1, filtering::filter_sharpen_u16_c1);

impl_directional_filter_image!(u16, C3, filtering::filter_sharpen_u16_c3);

impl_directional_filter_image!(u16, C4, filtering::filter_sharpen_u16_c4);

impl_directional_filter_image!(u16, AC4, filtering::filter_sharpen_u16_ac4);

impl_directional_filter_image!(i16, C1, filtering::filter_sharpen_i16_c1);

impl_directional_filter_image!(i16, C3, filtering::filter_sharpen_i16_c3);

impl_directional_filter_image!(i16, C4, filtering::filter_sharpen_i16_c4);

impl_directional_filter_image!(i16, AC4, filtering::filter_sharpen_i16_ac4);

impl_directional_filter_image!(f32, C1, filtering::filter_sharpen_f32_c1);

impl_directional_filter_image!(f32, C3, filtering::filter_sharpen_f32_c3);

impl_directional_filter_image!(f32, C4, filtering::filter_sharpen_f32_c4);

impl_directional_filter_image!(f32, AC4, filtering::filter_sharpen_f32_ac4);
