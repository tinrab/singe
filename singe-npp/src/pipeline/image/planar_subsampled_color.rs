use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        memory::Image,
        view::{C1, C2, ChannelLayout, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::planar::{SemiplanarImage, subsampled_size};

type Nv12PackedInverse<T, L> = for<'y, 'uv, 'destination> fn(
    &StreamContext,
    &ImageView<'y, T, C1>,
    &ImageView<'uv, T, C2>,
    &mut ImageViewMut<'destination, T, L>,
) -> Result<()>;

type Nv12PackedInverseColorTwist<T, L> = for<'y, 'uv, 'destination> fn(
    &StreamContext,
    &ImageView<'y, T, C1>,
    &ImageView<'uv, T, C2>,
    &mut ImageViewMut<'destination, T, L>,
    color::ColorTwistMatrix,
) -> Result<()>;

#[path = "planar_semiplanar_packed_color.rs"]
mod semiplanar_packed_color;

impl<T> SemiplanarImage<T>
where
    T: Copy,
{
    pub fn create_nv12(size: Size) -> Result<Self>
    where
        Workspace: ImageAllocator<T, C1> + ImageAllocator<T, C2>,
    {
        Self::create(size, 2, 2)
    }

    pub fn create(
        size: Size,
        horizontal_subsampling: i32,
        vertical_subsampling: i32,
    ) -> Result<Self>
    where
        Workspace: ImageAllocator<T, C1> + ImageAllocator<T, C2>,
    {
        Ok(Self {
            y: Workspace::create_image(size)?,
            uv: Workspace::create_image(subsampled_size(
                size,
                horizontal_subsampling,
                vertical_subsampling,
            )?)?,
        })
    }

    fn color_convert_to_packed<L>(
        self,
        stream_context: &StreamContext,
        operation: Nv12PackedInverse<T, L>,
    ) -> Result<Image<T, L>>
    where
        L: ChannelLayout,
        Workspace: ImageAllocator<T, L>,
    {
        let mut destination = Workspace::create_image(self.y.size())?;

        {
            let y = self.y.view()?;
            let uv = self.uv.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(stream_context, &y, &uv, &mut destination_view)?;
        }

        Ok(destination)
    }

    fn color_twist_to_packed<L>(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
        operation: Nv12PackedInverseColorTwist<T, L>,
    ) -> Result<Image<T, L>>
    where
        L: ChannelLayout,
        Workspace: ImageAllocator<T, L>,
    {
        let mut destination = Workspace::create_image(self.y.size())?;

        {
            let y = self.y.view()?;
            let uv = self.uv.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(stream_context, &y, &uv, &mut destination_view, twist)?;
        }

        Ok(destination)
    }
}
