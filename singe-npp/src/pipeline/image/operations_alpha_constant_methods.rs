use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::AlphaOperation,
};

use super::{AlphaCompConstantImage, ImageBacking, ImagePipeline};

#[path = "operations_alpha_premultiply_constant_methods.rs"]
mod premultiply_constant_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: AlphaCompConstantImage<T, L>,
    <Self as AlphaCompConstantImage<T, L>>::Alpha: Copy,
{
    pub fn alpha_comp_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        alpha: <Self as AlphaCompConstantImage<T, L>>::Alpha,
        other: &ImageView<'_, T, L>,
        other_alpha: <Self as AlphaCompConstantImage<T, L>>::Alpha,
        destination: &mut ImageViewMut<'_, T, L>,
        operation: AlphaOperation,
    ) -> Result<()> {
        <Self as AlphaCompConstantImage<T, L>>::alpha_comp_constant_image(
            stream_context,
            source,
            alpha,
            other,
            other_alpha,
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
    Self: AlphaCompConstantImage<T, L>,
    <Self as AlphaCompConstantImage<T, L>>::Alpha: Copy,
{
    pub fn alpha_comp_constant(
        self,
        alpha: <Self as AlphaCompConstantImage<T, L>>::Alpha,
        other: &ImageView<'_, T, L>,
        other_alpha: <Self as AlphaCompConstantImage<T, L>>::Alpha,
        operation: AlphaOperation,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as AlphaCompConstantImage<T, L>>::alpha_comp_constant_image(
                self.stream_context,
                &source,
                alpha,
                other,
                other_alpha,
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
