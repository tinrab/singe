use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
};

pub trait ExtractChannelImage<T, L> {
    fn extract_channel_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, C1>,
        channel: usize,
    ) -> Result<()>;
}

pub trait InsertChannelImage<T, L> {
    fn insert_channel_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, L>,
        channel: usize,
    ) -> Result<()>;
}

pub trait SetChannelImage<T, L> {
    fn set_channel_image(
        stream_context: &StreamContext,
        value: T,
        destination: &mut ImageViewMut<'_, T, L>,
        channel: usize,
    ) -> Result<()>;
}

pub trait CopyChannelImage<T, L> {
    fn copy_channel_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_channel: usize,
        destination: &mut ImageViewMut<'_, T, L>,
        destination_channel: usize,
    ) -> Result<()>;
}
