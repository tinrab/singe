use crate::{
    error::Result,
    image::view::C1,
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point, Size},
};

use super::super::morphology_traits::{
    GrayMaskMorphologyBorderImage, GrayMaskMorphologyBorderOperation,
};
use super::{ImageBacking, ImagePipeline};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Workspace: ImageAllocator<T, C1>,
{
    pub fn gray_dilate_border<M>(
        self,
        source_offset: Point,
        mask: &[M],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self>
    where
        M: Copy,
        Self: GrayMaskMorphologyBorderImage<T, M>,
    {
        self.gray_morphology_border(
            source_offset,
            mask,
            mask_size,
            anchor,
            border_type,
            <Self as GrayMaskMorphologyBorderImage<T, M>>::gray_dilate_border_image,
        )
    }

    pub fn gray_erode_border<M>(
        self,
        source_offset: Point,
        mask: &[M],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self>
    where
        M: Copy,
        Self: GrayMaskMorphologyBorderImage<T, M>,
    {
        self.gray_morphology_border(
            source_offset,
            mask,
            mask_size,
            anchor,
            border_type,
            <Self as GrayMaskMorphologyBorderImage<T, M>>::gray_erode_border_image,
        )
    }

    fn gray_morphology_border<M>(
        self,
        source_offset: Point,
        mask: &[M],
        mask_size: Size,
        anchor: Point,
        border_type: BorderType,
        operation: GrayMaskMorphologyBorderOperation<T, M>,
    ) -> Result<Self>
    where
        M: Copy,
    {
        let mut destination = self.workspace.image::<T, C1>(self.size())?;

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
