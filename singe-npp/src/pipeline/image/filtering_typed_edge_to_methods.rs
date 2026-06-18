use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::MaskSize,
};

use super::super::{ImagePipeline, filtering::*};

#[path = "filtering_typed_edge_border_to_methods.rs"]
mod border_to_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn filter_sobel_horizontal_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_sobel_horizontal_to_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }

    pub fn filter_sobel_vertical_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_sobel_vertical_to_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }

    pub fn filter_scharr_horizontal_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, M>,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_scharr_horizontal_to_image(
            stream_context,
            source,
            destination,
        )
    }

    pub fn filter_scharr_vertical_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, D, M>,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_scharr_vertical_to_image(
            stream_context,
            source,
            destination,
        )
    }
}
