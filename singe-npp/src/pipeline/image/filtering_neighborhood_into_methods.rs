use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::{BorderType, Point, Size},
};

use super::super::{ImagePipeline, filtering::*};

#[path = "filtering_neighborhood_border_into_methods.rs"]
mod border_into_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MedianFilterImage<T, L>,
{
    pub fn filter_median_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()> {
        <Self as MedianFilterImage<T, L>>::filter_median_image(
            stream_context,
            source,
            destination,
            mask_size,
            anchor,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: NeighborhoodFilterImage<T, L>,
{
    pub fn filter_max_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()> {
        <Self as NeighborhoodFilterImage<T, L>>::filter_max_image(
            stream_context,
            source,
            destination,
            mask_size,
            anchor,
        )
    }

    pub fn filter_min_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()> {
        <Self as NeighborhoodFilterImage<T, L>>::filter_min_image(
            stream_context,
            source,
            destination,
            mask_size,
            anchor,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: NeighborhoodBorderFilterImage<T, L>,
{
    pub fn filter_max_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as NeighborhoodBorderFilterImage<T, L>>::filter_max_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            anchor,
            border_type,
        )
    }

    pub fn filter_min_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as NeighborhoodBorderFilterImage<T, L>>::filter_min_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            anchor,
            border_type,
        )
    }
}
