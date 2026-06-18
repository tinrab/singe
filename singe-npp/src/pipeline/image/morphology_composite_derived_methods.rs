use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point, Size},
};

use super::super::super::{ImagePipeline, morphology_traits::CompositeMorphologyBorderImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CompositeMorphologyBorderImage<T, L>,
{
    pub fn morph_top_hat_border(
        self,
        source_offset: Point,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.composite_morphology_border(
            source_offset,
            mask,
            mask_size,
            anchor,
            border_type,
            <Self as CompositeMorphologyBorderImage<T, L>>::morph_top_hat_border_image,
        )
    }

    pub fn morph_black_hat_border(
        self,
        source_offset: Point,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.composite_morphology_border(
            source_offset,
            mask,
            mask_size,
            anchor,
            border_type,
            <Self as CompositeMorphologyBorderImage<T, L>>::morph_black_hat_border_image,
        )
    }

    pub fn morph_gradient_border(
        self,
        source_offset: Point,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.composite_morphology_border(
            source_offset,
            mask,
            mask_size,
            anchor,
            border_type,
            <Self as CompositeMorphologyBorderImage<T, L>>::morph_gradient_border_image,
        )
    }
}
