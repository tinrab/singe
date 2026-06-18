use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::{BorderType, MaskSize, Point},
};

use super::super::super::{ImagePipeline, filtering::TypedEdgeDirectionalBorderFilterImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn filter_sobel_horizontal_border_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalBorderFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalBorderFilterImage<
            T,
            L,
            D,
            M,
        >>::filter_sobel_horizontal_border_to_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }

    pub fn filter_sobel_vertical_border_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalBorderFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalBorderFilterImage<
            T,
            L,
            D,
            M,
        >>::filter_sobel_vertical_border_to_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }

    pub fn filter_scharr_horizontal_border_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, D, M>,
        border_type: BorderType,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalBorderFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalBorderFilterImage<
            T,
            L,
            D,
            M,
        >>::filter_scharr_horizontal_border_to_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }

    pub fn filter_scharr_vertical_border_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, D, M>,
        border_type: BorderType,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalBorderFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalBorderFilterImage<
            T,
            L,
            D,
            M,
        >>::filter_scharr_vertical_border_to_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}
