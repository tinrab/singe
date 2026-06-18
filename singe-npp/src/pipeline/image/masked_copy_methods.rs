use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut, MaskView},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::{CopyImage, ImageBacking, ImagePipeline};

pub trait MaskedCopyImage<T, L> {
    fn copy_masked_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<()>;
}

macro_rules! impl_masked_copy_image {
    ($ty:ty, $layout:ty, $copy:path) => {
        impl<'a> MaskedCopyImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn copy_masked_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask: &MaskView<'_>,
            ) -> Result<()> {
                $copy(stream_context, source, destination, mask)
            }
        }
    };
}

impl_masked_copy_image!(u8, C1, exchange::copy_masked_c1);
impl_masked_copy_image!(u8, C3, exchange::copy_masked_c3);
impl_masked_copy_image!(u8, C4, exchange::copy_masked_c4);
impl_masked_copy_image!(u8, AC4, exchange::copy_masked_ac4);
impl_masked_copy_image!(u16, C1, exchange::copy_masked_c1);
impl_masked_copy_image!(u16, C3, exchange::copy_masked_c3);
impl_masked_copy_image!(u16, C4, exchange::copy_masked_c4);
impl_masked_copy_image!(u16, AC4, exchange::copy_masked_ac4);
impl_masked_copy_image!(i16, C1, exchange::copy_masked_c1);
impl_masked_copy_image!(i16, C3, exchange::copy_masked_c3);
impl_masked_copy_image!(i16, C4, exchange::copy_masked_c4);
impl_masked_copy_image!(i16, AC4, exchange::copy_masked_ac4);
impl_masked_copy_image!(i32, C1, exchange::copy_masked_c1);
impl_masked_copy_image!(i32, C3, exchange::copy_masked_c3);
impl_masked_copy_image!(i32, C4, exchange::copy_masked_c4);
impl_masked_copy_image!(i32, AC4, exchange::copy_masked_ac4);
impl_masked_copy_image!(f32, C1, exchange::copy_masked_c1);
impl_masked_copy_image!(f32, C3, exchange::copy_masked_c3);
impl_masked_copy_image!(f32, C4, exchange::copy_masked_c4);
impl_masked_copy_image!(f32, AC4, exchange::copy_masked_ac4);

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: crate::image::view::ChannelLayout,
    Self: MaskedCopyImage<T, L>,
{
    pub fn copy_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<()> {
        <Self as MaskedCopyImage<T, L>>::copy_masked_image(
            stream_context,
            source,
            destination,
            mask,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: crate::image::view::ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CopyImage<T, L> + MaskedCopyImage<T, L>,
{
    pub fn copy_masked_from(
        mut self,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as MaskedCopyImage<T, L>>::copy_masked_image(
                    self.stream_context,
                    source,
                    &mut image_view,
                    mask,
                )?;
            }
            ImageBacking::Borrowed(destination_source) => {
                let mut destination = self.workspace.image::<T, L>(destination_source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as CopyImage<T, L>>::copy(
                    self.stream_context,
                    destination_source,
                    &mut destination_view,
                )?;
                <Self as MaskedCopyImage<T, L>>::copy_masked_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    mask,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
