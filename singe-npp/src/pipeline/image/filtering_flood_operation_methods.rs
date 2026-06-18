use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{ConnectedRegion, ImageNormalization, Point},
};

use super::{CopyImage, ImageBacking, ImagePipeline, filtering::FloodFillImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CopyImage<T, L> + FloodFillImage<T, L>,
    <Self as FloodFillImage<T, L>>::Value: Copy,
{
    pub(super) fn flood_fill_operation(
        self,
        seed: Point,
        new_value: <Self as FloodFillImage<T, L>>::Value,
        min: Option<<Self as FloodFillImage<T, L>>::Value>,
        max: Option<<Self as FloodFillImage<T, L>>::Value>,
        boundary_value: Option<<Self as FloodFillImage<T, L>>::Value>,
        norm: ImageNormalization,
        connected_region: Option<&mut ConnectedRegion>,
        operation: fn(
            &StreamContext,
            &mut ImageViewMut<'_, T, L>,
            Point,
            <Self as FloodFillImage<T, L>>::Value,
            Option<<Self as FloodFillImage<T, L>>::Value>,
            Option<<Self as FloodFillImage<T, L>>::Value>,
            Option<<Self as FloodFillImage<T, L>>::Value>,
            ImageNormalization,
            Option<&mut ConnectedRegion>,
        ) -> Result<()>,
    ) -> Result<Self> {
        let ImagePipeline {
            stream_context,
            workspace,
            backing,
        } = self;
        let mut image = match backing {
            ImageBacking::Owned(image) => image,
            ImageBacking::Borrowed(source) => {
                let mut destination = workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as CopyImage<T, L>>::copy(stream_context, &source, &mut destination_view)?;
                destination
            }
        };

        {
            let mut image_view = image.view_mut()?;
            operation(
                stream_context,
                &mut image_view,
                seed,
                new_value,
                min,
                max,
                boundary_value,
                norm,
                connected_region,
            )?;
        }

        Ok(Self {
            stream_context,
            workspace,
            backing: ImageBacking::Owned(image),
        })
    }
}
