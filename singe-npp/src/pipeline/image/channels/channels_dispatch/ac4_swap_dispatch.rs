use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, ImageOutputView, ImageView},
    },
    pipeline::image::channels::{ImagePipeline, SwapChannelsAc4Image},
};

macro_rules! impl_swap_channels_ac4_image {
    ($ty:ty, $swap:path) => {
        unsafe impl<'a> SwapChannelsAc4Image<$ty> for ImagePipeline<'a, $ty, AC4> {
            fn swap_channels_ac4_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, AC4>,
                destination: &mut ImageOutputView<'_, $ty, AC4>,
                order: [i32; 3],
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $swap(stream_context, source, destination, order) })
            }
        }
    };
}

impl_swap_channels_ac4_image!(u8, exchange::swap_channels_ac4);

impl_swap_channels_ac4_image!(u16, exchange::swap_channels_ac4);

impl_swap_channels_ac4_image!(i16, exchange::swap_channels_ac4);

impl_swap_channels_ac4_image!(i32, exchange::swap_channels_ac4);

impl_swap_channels_ac4_image!(f32, exchange::swap_channels_ac4);
