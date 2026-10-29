use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{C3, C4, ImageOutputView, ImageView},
    },
    pipeline::image::channels::{CopyChannelImage, ImagePipeline},
};

macro_rules! impl_copy_channel_image {
    ($ty:ty, $layout:ty, $copy:path) => {
        unsafe impl<'a> CopyChannelImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn copy_channel_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_channel: usize,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                destination_channel: usize,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $copy(
                        stream_context,
                        source,
                        source_channel,
                        destination,
                        destination_channel,
                    )
                })
            }
        }
    };
}

impl_copy_channel_image!(u8, C3, exchange::copy_channel_c3);

impl_copy_channel_image!(u8, C4, exchange::copy_channel_c4);

impl_copy_channel_image!(u16, C3, exchange::copy_channel_c3);

impl_copy_channel_image!(u16, C4, exchange::copy_channel_c4);

impl_copy_channel_image!(i16, C3, exchange::copy_channel_c3);

impl_copy_channel_image!(i16, C4, exchange::copy_channel_c4);

impl_copy_channel_image!(i32, C3, exchange::copy_channel_c3);

impl_copy_channel_image!(i32, C4, exchange::copy_channel_c4);

impl_copy_channel_image!(f32, C3, exchange::copy_channel_c3);

impl_copy_channel_image!(f32, C4, exchange::copy_channel_c4);
