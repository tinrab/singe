use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{C3, C4, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline};

pub trait SwapChannelsC3ToC4Image<T> {
    fn swap_channels_c3_to_c4_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        destination: &mut ImageViewMut<'_, T, C4>,
        order: [i32; 4],
        value: T,
    ) -> Result<()>;
}

macro_rules! impl_swap_channels_c3_to_c4_image {
    ($ty:ty, $swap:path) => {
        impl<'a> SwapChannelsC3ToC4Image<$ty> for ImagePipeline<'a, $ty, C3> {
            fn swap_channels_c3_to_c4_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C3>,
                destination: &mut ImageViewMut<'_, $ty, C4>,
                order: [i32; 4],
                value: $ty,
            ) -> Result<()> {
                $swap(stream_context, source, destination, order, value)
            }
        }
    };
}

impl_swap_channels_c3_to_c4_image!(u8, exchange::swap_channels_c3_to_c4);
impl_swap_channels_c3_to_c4_image!(u16, exchange::swap_channels_c3_to_c4);
impl_swap_channels_c3_to_c4_image!(i16, exchange::swap_channels_c3_to_c4);
impl_swap_channels_c3_to_c4_image!(i32, exchange::swap_channels_c3_to_c4);
impl_swap_channels_c3_to_c4_image!(f32, exchange::swap_channels_c3_to_c4);

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy,
    Self: SwapChannelsC3ToC4Image<T>,
{
    pub fn swap_channels_to_c4_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        destination: &mut ImageViewMut<'_, T, C4>,
        order: [i32; 4],
        value: T,
    ) -> Result<()> {
        <Self as SwapChannelsC3ToC4Image<T>>::swap_channels_c3_to_c4_image(
            stream_context,
            source,
            destination,
            order,
            value,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C3>
where
    T: Copy,
    Workspace: ImageAllocator<T, C4>,
    Self: SwapChannelsC3ToC4Image<T>,
{
    pub fn swap_channels_to_c4(
        self,
        order: [i32; 4],
        value: T,
    ) -> Result<ImagePipeline<'a, T, C4>> {
        let mut destination = self.workspace.image::<T, C4>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as SwapChannelsC3ToC4Image<T>>::swap_channels_c3_to_c4_image(
                self.stream_context,
                &source,
                &mut destination_view,
                order,
                value,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
