use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView},
};

/// # Safety
/// Complete borrowed native work on every path and uphold the operation's declared write coverage.
pub unsafe trait ExtractChannelImage<T, L> {
    fn extract_channel_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, C1>,
        channel: usize,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the operation's declared write coverage.
pub unsafe trait InsertChannelImage<T, L> {
    fn insert_channel_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, T, L>,
        channel: usize,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the operation's declared write coverage.
pub unsafe trait SetChannelImage<T, L> {
    fn set_channel_image(
        stream_context: &StreamContext,
        value: T,
        destination: &mut ImageOutputView<'_, T, L>,
        channel: usize,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the operation's declared write coverage.
pub unsafe trait CopyChannelImage<T, L> {
    fn copy_channel_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_channel: usize,
        destination: &mut ImageOutputView<'_, T, L>,
        destination_channel: usize,
    ) -> Result<()>;
}
