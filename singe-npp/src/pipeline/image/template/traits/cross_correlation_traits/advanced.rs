use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageOutputView, ImageView},
};

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait CrossCorrelationNormLevelAdvancedImage<T, L> {
    fn full(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;

    fn same(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;

    fn valid(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait CrossCorrelationNormLevelAdvancedToImage<T, D, L> {
    fn full(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, L>,
    ) -> Result<()>;

    fn same(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, L>,
    ) -> Result<()>;

    fn valid(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        template: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, L>,
    ) -> Result<()>;
}
