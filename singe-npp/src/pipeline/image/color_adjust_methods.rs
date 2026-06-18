use crate::{
    context::StreamContext,
    error::Result,
    image::view::{
        C1, ChannelLayout, ImageView, ImageViewMut, PlanarImageView, PlanarImageViewMut,
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::color::{ColorTwistImage, PlanarColorTwistImage};
use super::{ColorTwistMatrix, ImageBacking, ImagePipeline};

#[path = "color_adjust_batch_methods.rs"]
mod batch_methods;
#[path = "color_adjust_constants_methods.rs"]
mod constants_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ColorTwistImage<T, L>,
{
    pub fn color_twist_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        twist: ColorTwistMatrix,
    ) -> Result<()> {
        <Self as ColorTwistImage<T, L>>::color_twist_image(
            stream_context,
            source,
            destination,
            twist,
        )
    }

    pub fn color_twist_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        twist: ColorTwistMatrix,
    ) -> Result<()> {
        <Self as ColorTwistImage<T, L>>::color_twist_image_in_place(stream_context, image, twist)
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Self: PlanarColorTwistImage<T>,
{
    pub fn color_twist_planar_into(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, T, 3>,
        destination: &mut PlanarImageViewMut<'_, T, 3>,
        twist: ColorTwistMatrix,
    ) -> Result<()> {
        <Self as PlanarColorTwistImage<T>>::color_twist_planar_image(
            stream_context,
            source,
            destination,
            twist,
        )
    }

    pub fn color_twist_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, T, 3>,
        twist: ColorTwistMatrix,
    ) -> Result<()> {
        <Self as PlanarColorTwistImage<T>>::color_twist_planar_image_in_place(
            stream_context,
            image,
            twist,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: ColorTwistImage<T, L>,
{
    pub fn color_twist(mut self, twist: ColorTwistMatrix) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as ColorTwistImage<T, L>>::color_twist_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    twist,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as ColorTwistImage<T, L>>::color_twist_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    twist,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
