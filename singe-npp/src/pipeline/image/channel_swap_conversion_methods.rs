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

#[path = "channel_swap_to_c4_methods.rs"]
mod to_c4_methods;

pub trait SwapChannelsC4ToC3Image<T> {
    fn swap_channels_c4_to_c3_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C4>,
        destination: &mut ImageViewMut<'_, T, C3>,
        order: [i32; 3],
    ) -> Result<()>;
}

macro_rules! impl_swap_channels_c4_to_c3_image {
    ($ty:ty, $swap:path) => {
        impl<'a> SwapChannelsC4ToC3Image<$ty> for ImagePipeline<'a, $ty, C4> {
            fn swap_channels_c4_to_c3_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C4>,
                destination: &mut ImageViewMut<'_, $ty, C3>,
                order: [i32; 3],
            ) -> Result<()> {
                $swap(stream_context, source, destination, order)
            }
        }
    };
}

impl_swap_channels_c4_to_c3_image!(u8, exchange::swap_channels_c4_to_c3);
impl_swap_channels_c4_to_c3_image!(u16, exchange::swap_channels_c4_to_c3);
impl_swap_channels_c4_to_c3_image!(i16, exchange::swap_channels_c4_to_c3);
impl_swap_channels_c4_to_c3_image!(i32, exchange::swap_channels_c4_to_c3);
impl_swap_channels_c4_to_c3_image!(f32, exchange::swap_channels_c4_to_c3);

impl<'a, T> ImagePipeline<'a, T, C4>
where
    T: Copy,
    Self: SwapChannelsC4ToC3Image<T>,
{
    pub fn swap_channels_to_c3_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C4>,
        destination: &mut ImageViewMut<'_, T, C3>,
        order: [i32; 3],
    ) -> Result<()> {
        <Self as SwapChannelsC4ToC3Image<T>>::swap_channels_c4_to_c3_image(
            stream_context,
            source,
            destination,
            order,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C4>
where
    T: Copy,
    Workspace: ImageAllocator<T, C3>,
    Self: SwapChannelsC4ToC3Image<T>,
{
    pub fn swap_channels_to_c3(self, order: [i32; 3]) -> Result<ImagePipeline<'a, T, C3>> {
        let mut destination = self.workspace.image::<T, C3>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as SwapChannelsC4ToC3Image<T>>::swap_channels_c4_to_c3_image(
                self.stream_context,
                &source,
                &mut destination_view,
                order,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
