use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageOutputView, ImageView},
    types::{BorderType, Point, Size},
    workspace::ScratchBuffer,
};

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait CompositeMorphologyBorderImage<T, L> {
    fn morph_close_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;

    fn morph_open_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;

    fn morph_top_hat_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;

    fn morph_black_hat_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;

    fn morph_gradient_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
pub unsafe trait CompositeMorphologyBorderScratchImage<T, L> {
    fn morph_close_border_image_with_scratch(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        scratch: &mut ScratchBuffer,
        border_type: BorderType,
    ) -> Result<()>;

    fn morph_open_border_image_with_scratch(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        scratch: &mut ScratchBuffer,
        border_type: BorderType,
    ) -> Result<()>;

    fn morph_top_hat_border_image_with_scratch(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        scratch: &mut ScratchBuffer,
        border_type: BorderType,
    ) -> Result<()>;

    fn morph_black_hat_border_image_with_scratch(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        scratch: &mut ScratchBuffer,
        border_type: BorderType,
    ) -> Result<()>;

    fn morph_gradient_border_image_with_scratch(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        scratch: &mut ScratchBuffer,
        border_type: BorderType,
    ) -> Result<()>;
}
