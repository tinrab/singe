use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
    types::{BorderType, Point, Size},
    workspace::ScratchBuffer,
};

pub trait CompositeMorphologyBorderImage<T, L> {
    fn morph_close_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;

    fn morph_open_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;

    fn morph_top_hat_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;

    fn morph_black_hat_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;

    fn morph_gradient_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait CompositeMorphologyBorderScratchImage<T, L> {
    fn morph_close_border_image_with_scratch(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
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
        destination: &mut ImageViewMut<'_, T, L>,
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
        destination: &mut ImageViewMut<'_, T, L>,
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
        destination: &mut ImageViewMut<'_, T, L>,
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
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        scratch: &mut ScratchBuffer,
        border_type: BorderType,
    ) -> Result<()>;
}
