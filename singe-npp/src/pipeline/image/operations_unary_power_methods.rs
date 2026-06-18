use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{
    ImageBacking, ImagePipeline, operation_shapes::*, operation_traits::SquareImage,
};

#[path = "operations_square_root_methods.rs"]
mod square_root_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SquareImage<T, L>,
{
    pub fn square_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as SquareImage<T, L>>::square_image(stream_context, source, destination, scale_factor)
    }

    pub fn square_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as SquareImage<T, L>>::square_image_in_place(
            stream_context,
            source_destination,
            scale_factor,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: SquareImage<T, L>,
{
    pub fn square(self, scale_factor: i32) -> Result<Self> {
        unary_power_image(
            self,
            scale_factor,
            <Self as SquareImage<T, L>>::square_image,
            <Self as SquareImage<T, L>>::square_image_in_place,
        )
    }
}

fn unary_power_image<'a, T, L>(
    mut pipeline: ImagePipeline<'a, T, L>,
    scale_factor: i32,
    operation: ScaledUnaryPowerImageOperation<T, L>,
    operation_in_place: for<'destination> fn(
        &StreamContext,
        &mut ImageViewMut<'destination, T, L>,
        i32,
    ) -> Result<()>,
) -> Result<ImagePipeline<'a, T, L>>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
{
    match &mut pipeline.backing {
        ImageBacking::Owned(image) => {
            let mut image_view = image.view_mut()?;
            operation_in_place(pipeline.stream_context, &mut image_view, scale_factor)?;
        }
        ImageBacking::Borrowed(source) => {
            let mut destination = pipeline.workspace.image::<T, L>(source.size())?;
            let mut destination_view = destination.view_mut()?;
            operation(
                pipeline.stream_context,
                source,
                &mut destination_view,
                scale_factor,
            )?;
            pipeline.backing = ImageBacking::Owned(destination);
        }
    }

    Ok(pipeline)
}
