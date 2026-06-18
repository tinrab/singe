use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
    types::{BorderType, MaskSize, Point},
};

pub trait TypedSobelExtendedFilterImage<T, L, D, M> {
    fn filter_sobel_horizontal_second_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_sobel_vertical_second_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_sobel_cross_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_laplace_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

pub trait TypedSobelExtendedBorderFilterImage<T, L, D, M> {
    fn filter_sobel_horizontal_second_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_sobel_vertical_second_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_sobel_cross_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_laplace_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}
