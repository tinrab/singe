use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline};

pub trait SetImage<T, L> {
    type Value;

    fn set_image(
        stream_context: &StreamContext,
        value: Self::Value,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

macro_rules! impl_set_image {
    ($ty:ty, $layout:ty, $value_ty:ty, $set:path) => {
        impl<'a> SetImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Value = $value_ty;

            fn set_image(
                stream_context: &StreamContext,
                value: Self::Value,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $set(stream_context, value, destination)
            }
        }
    };
}

#[path = "fill_dispatch_impls.rs"]
mod dispatch_impls;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SetImage<T, L>,
    <Self as SetImage<T, L>>::Value: Copy,
{
    pub fn set_into(
        stream_context: &StreamContext,
        value: <Self as SetImage<T, L>>::Value,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as SetImage<T, L>>::set_image(stream_context, value, destination)
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: SetImage<T, L>,
    <Self as SetImage<T, L>>::Value: Copy,
{
    pub fn set(mut self, value: <Self as SetImage<T, L>>::Value) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as SetImage<T, L>>::set_image(self.stream_context, value, &mut image_view)?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as SetImage<T, L>>::set_image(
                    self.stream_context,
                    value,
                    &mut destination_view,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
