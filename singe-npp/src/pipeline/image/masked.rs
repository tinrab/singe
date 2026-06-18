use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{AC4, C1, C3, C4, ChannelLayout, ImageViewMut, MaskView},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::{CopyImage, ImageBacking, ImagePipeline};

pub trait MaskedSetImage<T, L> {
    type Value;

    fn set_masked_image(
        stream_context: &StreamContext,
        value: Self::Value,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<()>;
}

macro_rules! impl_masked_set_image {
    ($ty:ty, $layout:ty, $value_ty:ty, $set:path) => {
        impl<'a> MaskedSetImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Value = $value_ty;

            fn set_masked_image(
                stream_context: &StreamContext,
                value: Self::Value,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask: &MaskView<'_>,
            ) -> Result<()> {
                $set(stream_context, value, destination, mask)
            }
        }
    };
}

impl_masked_set_image!(u8, C1, u8, exchange::set_masked_c1);
impl_masked_set_image!(u8, C3, [u8; 3], exchange::set_masked_c3);
impl_masked_set_image!(u8, C4, [u8; 4], exchange::set_masked_c4);
impl_masked_set_image!(u8, AC4, [u8; 3], exchange::set_masked_ac4);
impl_masked_set_image!(u16, C1, u16, exchange::set_masked_c1);
impl_masked_set_image!(u16, C3, [u16; 3], exchange::set_masked_c3);
impl_masked_set_image!(u16, C4, [u16; 4], exchange::set_masked_c4);
impl_masked_set_image!(u16, AC4, [u16; 3], exchange::set_masked_ac4);
impl_masked_set_image!(i16, C1, i16, exchange::set_masked_c1);
impl_masked_set_image!(i16, C3, [i16; 3], exchange::set_masked_c3);
impl_masked_set_image!(i16, C4, [i16; 4], exchange::set_masked_c4);
impl_masked_set_image!(i16, AC4, [i16; 3], exchange::set_masked_ac4);
impl_masked_set_image!(i32, C1, i32, exchange::set_masked_c1);
impl_masked_set_image!(i32, C3, [i32; 3], exchange::set_masked_c3);
impl_masked_set_image!(i32, C4, [i32; 4], exchange::set_masked_c4);
impl_masked_set_image!(i32, AC4, [i32; 3], exchange::set_masked_ac4);
impl_masked_set_image!(f32, C1, f32, exchange::set_masked_c1);
impl_masked_set_image!(f32, C3, [f32; 3], exchange::set_masked_c3);
impl_masked_set_image!(f32, C4, [f32; 4], exchange::set_masked_c4);
impl_masked_set_image!(f32, AC4, [f32; 3], exchange::set_masked_ac4);

#[path = "masked_copy_methods.rs"]
mod masked_copy_methods;

#[path = "masked_accumulate_methods.rs"]
mod masked_accumulate_methods;
#[path = "masked_accumulate_product_methods.rs"]
mod masked_accumulate_product_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedSetImage<T, L>,
    <Self as MaskedSetImage<T, L>>::Value: Copy,
{
    pub fn set_masked_into(
        stream_context: &StreamContext,
        value: <Self as MaskedSetImage<T, L>>::Value,
        destination: &mut ImageViewMut<'_, T, L>,
        mask: &MaskView<'_>,
    ) -> Result<()> {
        <Self as MaskedSetImage<T, L>>::set_masked_image(stream_context, value, destination, mask)
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: CopyImage<T, L> + MaskedSetImage<T, L>,
    <Self as MaskedSetImage<T, L>>::Value: Copy,
{
    pub fn set_masked(
        mut self,
        value: <Self as MaskedSetImage<T, L>>::Value,
        mask: &MaskView<'_>,
    ) -> Result<Self> {
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as MaskedSetImage<T, L>>::set_masked_image(
                    self.stream_context,
                    value,
                    &mut image_view,
                    mask,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = self.workspace.image::<T, L>(source.size())?;
                let mut destination_view = destination.view_mut()?;
                <Self as CopyImage<T, L>>::copy(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                <Self as MaskedSetImage<T, L>>::set_masked_image(
                    self.stream_context,
                    value,
                    &mut destination_view,
                    mask,
                )?;
                self.backing = ImageBacking::Owned(destination);
            }
        }

        Ok(self)
    }
}
