use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::{BorderType, Point, Size},
    workspace::ScratchBuffer,
};

use super::super::super::{
    ImagePipeline, morphology_traits::CompositeMorphologyBorderScratchImage,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn morph_top_hat_border_with_scratch_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        scratch: &mut ScratchBuffer,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: CompositeMorphologyBorderScratchImage<T, L>,
    {
        <Self as CompositeMorphologyBorderScratchImage<T, L>>::morph_top_hat_border_image_with_scratch(
            stream_context,
            source,
            source_offset,
            destination,
            mask,
            mask_size,
            anchor,
            scratch,
            border_type,
        )
    }

    pub fn morph_black_hat_border_with_scratch_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        scratch: &mut ScratchBuffer,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: CompositeMorphologyBorderScratchImage<T, L>,
    {
        <Self as CompositeMorphologyBorderScratchImage<T, L>>::morph_black_hat_border_image_with_scratch(
            stream_context,
            source,
            source_offset,
            destination,
            mask,
            mask_size,
            anchor,
            scratch,
            border_type,
        )
    }

    pub fn morph_gradient_border_with_scratch_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        scratch: &mut ScratchBuffer,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: CompositeMorphologyBorderScratchImage<T, L>,
    {
        <Self as CompositeMorphologyBorderScratchImage<T, L>>::morph_gradient_border_image_with_scratch(
            stream_context,
            source,
            source_offset,
            destination,
            mask,
            mask_size,
            anchor,
            scratch,
            border_type,
        )
    }
}
