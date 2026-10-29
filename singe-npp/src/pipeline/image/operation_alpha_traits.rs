use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, ImageOutputView, ImageView, ImageViewMut},
    types::AlphaOperation,
};

/// # Safety
/// Complete borrowed work on every path and write valid values to every active output channel.
/// Read-write operations require initialized inputs; AC4 dispatch preserves ignored alpha.
pub unsafe trait AlphaPremultiplyConstantImage<T, L> {
    type Alpha;

    fn alpha_premultiply_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        alpha: Self::Alpha,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;

    fn alpha_premultiply_constant_image_in_place(
        stream_context: &StreamContext,
        alpha: Self::Alpha,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed work on every path and write valid values to every active output channel.
/// Read-write operations require initialized inputs; AC4 dispatch preserves ignored alpha.
pub unsafe trait ColorKeyImage<L> {
    type ColorKey;

    fn comp_color_key_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, u8, L>,
        source2: &ImageView<'_, u8, L>,
        color_key: Self::ColorKey,
        destination: &mut ImageOutputView<'_, u8, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed work on every path and write valid values to every active output channel.
/// Read-write operations require initialized inputs; AC4 dispatch preserves ignored alpha.
pub unsafe trait AlphaCompConstantImage<T, L> {
    type Alpha;

    fn alpha_comp_constant_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, L>,
        alpha1: Self::Alpha,
        source2: &ImageView<'_, T, L>,
        alpha2: Self::Alpha,
        destination: &mut ImageOutputView<'_, T, L>,
        operation: AlphaOperation,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed work on every path and write valid values to every active output channel.
/// Read-write operations require initialized inputs; AC4 dispatch preserves ignored alpha.
pub unsafe trait AlphaCompImage<T, L> {
    fn alpha_comp_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        operation: AlphaOperation,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed work on every path and write valid values to every active output channel.
/// Read-write operations require initialized inputs; AC4 dispatch preserves ignored alpha.
pub unsafe trait AlphaPremultiplyImage<T> {
    fn alpha_premultiply_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, AC4>,
        destination: &mut ImageOutputView<'_, T, AC4>,
    ) -> Result<()>;

    fn alpha_premultiply_image_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, AC4>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed work on every path and write valid values to every active output channel.
/// Read-write operations require initialized inputs; AC4 dispatch preserves ignored alpha.
pub unsafe trait AlphaCompColorKeyImage {
    fn alpha_comp_color_key_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, u8, AC4>,
        alpha1: u8,
        source2: &ImageView<'_, u8, AC4>,
        alpha2: u8,
        color_key: [u8; 4],
        destination: &mut ImageOutputView<'_, u8, AC4>,
        operation: AlphaOperation,
    ) -> Result<()>;
}
