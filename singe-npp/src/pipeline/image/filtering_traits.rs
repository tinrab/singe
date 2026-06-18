use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
    types::{BorderType, Point, Size},
};

pub(super) use super::filtering_distance_traits::*;
pub(super) use super::filtering_edge_traits::*;
pub(super) use super::filtering_edge_typed_traits::*;
pub(super) use super::filtering_feature_traits::*;
pub(super) use super::filtering_flood_traits::*;
pub(super) use super::filtering_kernel_traits::*;
pub(super) use super::filtering_masked_traits::*;
pub(super) use super::filtering_segmentation_traits::*;
pub(super) use super::filtering_separable_traits::*;

pub trait BoxFilterImage<T, L> {
    fn filter_box_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>;
}

pub trait BoxBorderFilterImage<T, L> {
    fn filter_box_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait AdaptiveBoxThresholdBorderImage<T, L> {
    fn filter_threshold_adaptive_box_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        delta: f32,
        value_greater_than: T,
        value_less_or_equal: T,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait NeighborhoodFilterImage<T, L> {
    fn filter_max_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>;

    fn filter_min_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>;
}

pub trait NeighborhoodBorderFilterImage<T, L> {
    fn filter_max_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_min_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}

pub trait MedianFilterImage<T, L> {
    fn filter_median_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>;
}

pub trait MedianBorderFilterImage<T, L> {
    fn filter_median_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}
