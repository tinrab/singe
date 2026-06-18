use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::{BorderType, Point, Size},
};

use super::super::{
    ImagePipeline,
    morphology_traits::{MaskMorphologyBorderImage, MaskMorphologyImage},
};

#[path = "morphology_gray_border_into_methods.rs"]
mod gray_border_into_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn dilate_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>
    where
        Self: MaskMorphologyImage<T, L>,
    {
        <Self as MaskMorphologyImage<T, L>>::dilate_image(
            stream_context,
            source,
            destination,
            mask,
            mask_size,
            anchor,
        )
    }

    pub fn erode_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
    ) -> Result<()>
    where
        Self: MaskMorphologyImage<T, L>,
    {
        <Self as MaskMorphologyImage<T, L>>::erode_image(
            stream_context,
            source,
            destination,
            mask,
            mask_size,
            anchor,
        )
    }

    pub fn dilate_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: MaskMorphologyBorderImage<T, L>,
    {
        <Self as MaskMorphologyBorderImage<T, L>>::dilate_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask,
            mask_size,
            anchor,
            border_type,
        )
    }

    pub fn erode_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: MaskMorphologyBorderImage<T, L>,
    {
        <Self as MaskMorphologyBorderImage<T, L>>::erode_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask,
            mask_size,
            anchor,
            border_type,
        )
    }
}
