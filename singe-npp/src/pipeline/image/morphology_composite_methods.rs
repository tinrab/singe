use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point, Size},
};

use super::super::{
    ImageBacking, ImagePipeline,
    morphology_traits::{CompositeMorphologyBorderImage, MaskMorphologyBorderOperation},
};

#[path = "morphology_composite_derived_methods.rs"]
mod derived_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CompositeMorphologyBorderImage<T, L>,
{
    pub fn morph_close_border(
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
            <Self as CompositeMorphologyBorderImage<T, L>>::morph_close_border_image,
        )
    }

    pub fn morph_open_border(
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
            <Self as CompositeMorphologyBorderImage<T, L>>::morph_open_border_image,
        )
    }

    pub(super) fn composite_morphology_border(
        self,
        source_offset: Point,
        mask: &[u8],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
        operation: MaskMorphologyBorderOperation<T, L>,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask,
                mask_size,
                anchor,
                border_type,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
