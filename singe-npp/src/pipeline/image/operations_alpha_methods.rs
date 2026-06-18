use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::AlphaOperation,
};

use super::{ImageBacking, ImagePipeline, operation_traits::*};

#[path = "operations_alpha_constant_methods.rs"]
mod constant_methods;
#[path = "operations_alpha_premultiply_methods.rs"]
mod premultiply_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: AlphaCompImage<T, L>,
{
    pub fn alpha_comp_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        operation: AlphaOperation,
    ) -> Result<()> {
        <Self as AlphaCompImage<T, L>>::alpha_comp_image(
            stream_context,
            left,
            right,
            destination,
            operation,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: AlphaCompImage<T, L>,
{
    pub fn alpha_comp(
        self,
        other: &ImageView<'_, T, L>,
        operation: AlphaOperation,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as AlphaCompImage<T, L>>::alpha_comp_image(
                self.stream_context,
                &source,
                other,
                &mut destination_view,
                operation,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
