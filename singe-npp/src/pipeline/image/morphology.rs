use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
    types::{Point, Size},
};

use super::morphology_traits::MaskMorphologyImage;
use super::{ImageBacking, ImagePipeline};

#[path = "morphology_border_methods.rs"]
mod border_methods;
#[path = "morphology_composite_into_methods.rs"]
mod composite_into_methods;
#[path = "morphology_composite_methods.rs"]
mod composite_methods;
#[path = "morphology_gray_methods.rs"]
mod gray_methods;
#[path = "morphology_into_methods.rs"]
mod into_methods;
#[path = "morphology_3x3_methods.rs"]
mod three_by_three_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: MaskMorphologyImage<T, L>,
{
    pub fn dilate(self, mask: &[u8], mask_size: Size, anchor: Point) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as MaskMorphologyImage<T, L>>::dilate_image(
                self.stream_context,
                &source,
                &mut destination_view,
                mask,
                mask_size,
                anchor,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    pub fn erode(self, mask: &[u8], mask_size: Size, anchor: Point) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as MaskMorphologyImage<T, L>>::erode_image(
                self.stream_context,
                &source,
                &mut destination_view,
                mask,
                mask_size,
                anchor,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
