use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline};

#[path = "channels_duplicate_ac4_methods.rs"]
mod ac4_methods;
#[path = "channels_duplicate_c4_methods.rs"]
mod c4_methods;

pub trait DuplicateToC3Image<T> {
    fn duplicate_to_c3_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C3>,
    ) -> Result<()>;
}

pub trait DuplicateToC4Image<T> {
    fn duplicate_to_c4_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C4>,
    ) -> Result<()>;
}

pub trait DuplicateToAC4Image<T> {
    fn duplicate_to_ac4_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, AC4>,
    ) -> Result<()>;
}

macro_rules! impl_duplicate_to_c3_image {
    ($ty:ty, $duplicate:path) => {
        impl<'a> DuplicateToC3Image<$ty> for ImagePipeline<'a, $ty, C1> {
            fn duplicate_to_c3_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageViewMut<'_, $ty, C3>,
            ) -> Result<()> {
                $duplicate(stream_context, source, destination)
            }
        }
    };
}

macro_rules! impl_duplicate_to_c4_image {
    ($ty:ty, $duplicate:path) => {
        impl<'a> DuplicateToC4Image<$ty> for ImagePipeline<'a, $ty, C1> {
            fn duplicate_to_c4_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageViewMut<'_, $ty, C4>,
            ) -> Result<()> {
                $duplicate(stream_context, source, destination)
            }
        }
    };
}

macro_rules! impl_duplicate_to_ac4_image {
    ($ty:ty, $duplicate:path) => {
        impl<'a> DuplicateToAC4Image<$ty> for ImagePipeline<'a, $ty, C1> {
            fn duplicate_to_ac4_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, C1>,
                destination: &mut ImageViewMut<'_, $ty, AC4>,
            ) -> Result<()> {
                $duplicate(stream_context, source, destination)
            }
        }
    };
}

#[path = "channels_duplicate_dispatch_impls.rs"]
mod dispatch_impls;

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Self: DuplicateToC3Image<T>,
{
    pub fn duplicate_to_c3_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, T, C3>,
    ) -> Result<()> {
        <Self as DuplicateToC3Image<T>>::duplicate_to_c3_image(stream_context, source, destination)
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Workspace: ImageAllocator<T, C3>,
    Self: DuplicateToC3Image<T>,
{
    pub fn duplicate_to_c3(self) -> Result<ImagePipeline<'a, T, C3>> {
        let mut destination = self.workspace.image::<T, C3>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as DuplicateToC3Image<T>>::duplicate_to_c3_image(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
