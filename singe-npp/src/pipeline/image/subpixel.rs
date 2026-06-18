use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C3, C4, ChannelLayout, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::{ImageBacking, ImagePipeline};

pub trait CopySubpixelImage<T, L> {
    fn copy_subpixel_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        x_shift: f32,
        y_shift: f32,
    ) -> Result<()>;
}

macro_rules! impl_copy_subpixel_image {
    ($ty:ty, $layout:ty, $copy:path) => {
        impl<'a> CopySubpixelImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn copy_subpixel_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                x_shift: f32,
                y_shift: f32,
            ) -> Result<()> {
                $copy(stream_context, source, destination, x_shift, y_shift)
            }
        }
    };
}

impl_copy_subpixel_image!(u8, C1, exchange::copy_subpixel_c1);
impl_copy_subpixel_image!(u8, C3, exchange::copy_subpixel_c3);
impl_copy_subpixel_image!(u8, C4, exchange::copy_subpixel_c4);
impl_copy_subpixel_image!(u8, AC4, exchange::copy_subpixel_ac4);
impl_copy_subpixel_image!(u16, C1, exchange::copy_subpixel_c1);
impl_copy_subpixel_image!(u16, C3, exchange::copy_subpixel_c3);
impl_copy_subpixel_image!(u16, C4, exchange::copy_subpixel_c4);
impl_copy_subpixel_image!(u16, AC4, exchange::copy_subpixel_ac4);
impl_copy_subpixel_image!(i16, C1, exchange::copy_subpixel_c1);
impl_copy_subpixel_image!(i16, C3, exchange::copy_subpixel_c3);
impl_copy_subpixel_image!(i16, C4, exchange::copy_subpixel_c4);
impl_copy_subpixel_image!(i16, AC4, exchange::copy_subpixel_ac4);
impl_copy_subpixel_image!(i32, C1, exchange::copy_subpixel_c1);
impl_copy_subpixel_image!(i32, C3, exchange::copy_subpixel_c3);
impl_copy_subpixel_image!(i32, C4, exchange::copy_subpixel_c4);
impl_copy_subpixel_image!(i32, AC4, exchange::copy_subpixel_ac4);
impl_copy_subpixel_image!(f32, C1, exchange::copy_subpixel_c1);
impl_copy_subpixel_image!(f32, C3, exchange::copy_subpixel_c3);
impl_copy_subpixel_image!(f32, C4, exchange::copy_subpixel_c4);
impl_copy_subpixel_image!(f32, AC4, exchange::copy_subpixel_ac4);

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CopySubpixelImage<T, L>,
{
    pub fn copy_subpixel_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        x_shift: f32,
        y_shift: f32,
    ) -> Result<()> {
        <Self as CopySubpixelImage<T, L>>::copy_subpixel_image(
            stream_context,
            source,
            destination,
            x_shift,
            y_shift,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CopySubpixelImage<T, L>,
{
    pub fn copy_subpixel(self, size: Size, x_shift: f32, y_shift: f32) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(size)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as CopySubpixelImage<T, L>>::copy_subpixel_image(
                self.stream_context,
                &source,
                &mut destination_view,
                x_shift,
                y_shift,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
