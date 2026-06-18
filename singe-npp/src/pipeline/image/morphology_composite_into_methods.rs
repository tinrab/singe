use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    types::{BorderType, Point, Size},
    workspace::ScratchBuffer,
};

use super::super::{ImagePipeline, morphology_traits::CompositeMorphologyBorderScratchImage};

#[path = "morphology_composite_derived_into_methods.rs"]
mod derived_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn morph_close_border_with_scratch_into(
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
        <Self as CompositeMorphologyBorderScratchImage<T, L>>::morph_close_border_image_with_scratch(
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

    pub fn morph_open_border_with_scratch_into(
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
        <Self as CompositeMorphologyBorderScratchImage<T, L>>::morph_open_border_image_with_scratch(
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
