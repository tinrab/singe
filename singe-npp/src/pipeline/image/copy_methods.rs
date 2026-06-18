use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        memory::Image,
        view::{AC4, C1, C2, C3, C4, ChannelLayout, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CopyImage<T, L>,
{
    pub fn copy_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as CopyImage<T, L>>::copy(stream_context, source, destination)
    }

    pub fn copy(self) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as CopyImage<T, L>>::copy(self.stream_context, &source, &mut destination_view)?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }

    // TODO: call `stream.synchronize()`?
    pub fn finish(self) -> Result<Image<T, L>> {
        match self.backing {
            ImageBacking::Owned(image) => Ok(image),
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as CopyImage<T, L>>::copy(
                    self.stream_context,
                    &source,
                    &mut destination_view,
                )?;
                Ok(destination)
            }
        }
    }
}

pub trait CopyImage<T, L> {
    fn copy(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

macro_rules! impl_copy_image {
    ($ty:ty, $layout:ty, $copy:path) => {
        impl<'a> CopyImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn copy(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $copy(stream_context, source, destination)
            }
        }
    };
}

#[path = "copy_complex_impls.rs"]
mod complex_impls;

impl_copy_image!(u8, C1, exchange::copy_c1);
impl_copy_image!(u8, C3, exchange::copy_c3);
impl_copy_image!(u8, C4, exchange::copy_c4);
impl_copy_image!(u8, AC4, exchange::copy_ac4);
impl_copy_image!(i8, C1, exchange::copy_c1);
impl_copy_image!(i8, C2, exchange::copy_c2);
impl_copy_image!(i8, C3, exchange::copy_c3);
impl_copy_image!(i8, C4, exchange::copy_c4);
impl_copy_image!(i8, AC4, exchange::copy_ac4);
impl_copy_image!(u16, C1, exchange::copy_c1);
impl_copy_image!(u16, C3, exchange::copy_c3);
impl_copy_image!(u16, C4, exchange::copy_c4);
impl_copy_image!(u16, AC4, exchange::copy_ac4);
impl_copy_image!(f16, C1, exchange::copy_c1);
impl_copy_image!(f16, C3, exchange::copy_c3);
impl_copy_image!(f16, C4, exchange::copy_c4);
impl_copy_image!(i16, C1, exchange::copy_c1);
impl_copy_image!(i16, C3, exchange::copy_c3);
impl_copy_image!(i16, C4, exchange::copy_c4);
impl_copy_image!(i16, AC4, exchange::copy_ac4);
impl_copy_image!(u32, C1, exchange::copy_c1);
impl_copy_image!(u32, AC4, exchange::copy_ac4);
impl_copy_image!(i32, C1, exchange::copy_c1);
impl_copy_image!(i32, C3, exchange::copy_c3);
impl_copy_image!(i32, C4, exchange::copy_c4);
impl_copy_image!(i32, AC4, exchange::copy_ac4);
impl_copy_image!(f32, C1, exchange::copy_c1);
impl_copy_image!(f32, C3, exchange::copy_c3);
impl_copy_image!(f32, C4, exchange::copy_c4);
impl_copy_image!(f32, AC4, exchange::copy_ac4);
