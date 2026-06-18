use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C4, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{
    ColorTwistConstants4, ColorTwistMatrix4, ImageBacking, ImagePipeline,
    color::ColorTwistWithConstantsImage,
};

#[path = "color_adjust_batch_constants_methods.rs"]
mod batch_methods;

impl<'a, T> ImagePipeline<'a, T, C4>
where
    T: Copy,
    Self: ColorTwistWithConstantsImage<T>,
{
    pub fn color_twist_with_constants_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C4>,
        destination: &mut ImageViewMut<'_, T, C4>,
        twist: ColorTwistMatrix4,
        constants: ColorTwistConstants4,
    ) -> Result<()> {
        <Self as ColorTwistWithConstantsImage<T>>::color_twist_with_constants_image(
            stream_context,
            source,
            destination,
            twist,
            constants,
        )
    }

    pub fn color_twist_with_constants_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, C4>,
        twist: ColorTwistMatrix4,
        constants: ColorTwistConstants4,
    ) -> Result<()> {
        <Self as ColorTwistWithConstantsImage<T>>::color_twist_with_constants_image_in_place(
            stream_context,
            image,
            twist,
            constants,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C4>
where
    T: Copy,
    Workspace: ImageAllocator<T, C4>,
    Self: ColorTwistWithConstantsImage<T>,
{
    pub fn color_twist_with_constants(
        mut self,
        twist: ColorTwistMatrix4,
        constants: ColorTwistConstants4,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as ColorTwistWithConstantsImage<T>>::color_twist_with_constants_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    twist,
                    constants,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, C4>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as ColorTwistWithConstantsImage<T>>::color_twist_with_constants_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    twist,
                    constants,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
