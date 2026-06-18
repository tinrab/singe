use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
    types::{BorderType, MaskSize, Point},
};

pub trait TypedEdgeDirectionalFilterImage<T, L, D, M> {
    fn filter_sobel_horizontal_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_sobel_vertical_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_scharr_horizontal_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, M>,
    ) -> Result<()>;

    fn filter_scharr_vertical_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, M>,
    ) -> Result<()>;
}

pub trait TypedEdgeDirectionalBorderFilterImage<T, L, D, M> {
    fn filter_sobel_horizontal_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_sobel_vertical_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_scharr_horizontal_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, D, M>,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_scharr_vertical_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, D, M>,
        border_type: BorderType,
    ) -> Result<()>;
}
