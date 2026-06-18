use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
    types::{BorderType, MaskSize, Point},
};

#[path = "filtering_edge_typed_directional_traits.rs"]
mod directional_traits;
#[path = "filtering_edge_typed_extended_traits.rs"]
mod extended_traits;

pub use directional_traits::*;
pub use extended_traits::*;

pub trait SobelExtendedFilterImage<T, L> {
    fn filter_sobel_horizontal_mask_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_sobel_vertical_mask_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_sobel_horizontal_second_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_sobel_vertical_second_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_sobel_cross_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

pub trait SobelExtendedBorderFilterImage<T, L> {
    fn filter_sobel_horizontal_mask_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_sobel_vertical_mask_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_sobel_horizontal_second_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_sobel_vertical_second_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_sobel_cross_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait ScharrFilterImage<T, L> {
    fn filter_scharr_horizontal_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn filter_scharr_vertical_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait ScharrBorderFilterImage<T, L> {
    fn filter_scharr_horizontal_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_scharr_vertical_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()>;
}
