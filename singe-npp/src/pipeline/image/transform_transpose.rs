use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{C1, C3, C4, ChannelLayout, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::{ImageBacking, ImagePipeline};

pub trait TransposeImage<T, L> {
    fn transpose_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

macro_rules! impl_transpose_image {
    ($ty:ty, $layout:ty, $transpose:path) => {
        impl<'a> TransposeImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn transpose_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $transpose(stream_context, source, destination)
            }
        }
    };
}

impl_transpose_image!(u8, C1, exchange::transpose_c1);
impl_transpose_image!(u8, C3, exchange::transpose_c3);
impl_transpose_image!(u8, C4, exchange::transpose_c4);
impl_transpose_image!(u16, C1, exchange::transpose_c1);
impl_transpose_image!(u16, C3, exchange::transpose_c3);
impl_transpose_image!(u16, C4, exchange::transpose_c4);
impl_transpose_image!(i16, C1, exchange::transpose_c1);
impl_transpose_image!(i16, C3, exchange::transpose_c3);
impl_transpose_image!(i16, C4, exchange::transpose_c4);
impl_transpose_image!(i32, C1, exchange::transpose_c1);
impl_transpose_image!(i32, C3, exchange::transpose_c3);
impl_transpose_image!(i32, C4, exchange::transpose_c4);
impl_transpose_image!(f32, C1, exchange::transpose_c1);
impl_transpose_image!(f32, C3, exchange::transpose_c3);
impl_transpose_image!(f32, C4, exchange::transpose_c4);

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: TransposeImage<T, L>,
{
    pub fn transpose_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as TransposeImage<T, L>>::transpose_image(stream_context, source, destination)
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: TransposeImage<T, L>,
{
    pub fn transpose(self) -> Result<Self> {
        let size = self.size();
        let destination_size = Size {
            width: size.height,
            height: size.width,
        };
        let mut destination = self.workspace.image::<T, L>(destination_size)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as TransposeImage<T, L>>::transpose_image(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
