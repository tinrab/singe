use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry::{
            Resize, resize_f16_c1, resize_f16_c3, resize_f16_c4, resize_f32_ac4, resize_f32_c1,
            resize_f32_c3, resize_f32_c4, resize_i16_ac4, resize_i16_c1, resize_i16_c3,
            resize_i16_c4, resize_u8_ac4, resize_u8_c1, resize_u8_c3, resize_u8_c4, resize_u16_ac4,
            resize_u16_c1, resize_u16_c3, resize_u16_c4,
        },
        view::{AC4, C1, C3, C4, ChannelLayout, ImageView, ImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::{InterpolationMode, Size},
};

use super::{ImageBacking, ImagePipeline};

pub trait ResizeImage<T, L> {
    fn resize_image(
        stream_context: &StreamContext,
        resize: &Resize,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

macro_rules! impl_resize_image {
    ($ty:ty, $layout:ty, $resize:path) => {
        impl<'a> ResizeImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn resize_image(
                stream_context: &StreamContext,
                resize: &Resize,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $resize(stream_context, resize, source, destination)
            }
        }
    };
}

impl_resize_image!(u8, C1, resize_u8_c1);
impl_resize_image!(u8, C3, resize_u8_c3);
impl_resize_image!(u8, C4, resize_u8_c4);
impl_resize_image!(u8, AC4, resize_u8_ac4);
impl_resize_image!(u16, C1, resize_u16_c1);
impl_resize_image!(u16, C3, resize_u16_c3);
impl_resize_image!(u16, C4, resize_u16_c4);
impl_resize_image!(u16, AC4, resize_u16_ac4);
impl_resize_image!(i16, C1, resize_i16_c1);
impl_resize_image!(i16, C3, resize_i16_c3);
impl_resize_image!(i16, C4, resize_i16_c4);
impl_resize_image!(i16, AC4, resize_i16_ac4);
impl_resize_image!(f16, C1, resize_f16_c1);
impl_resize_image!(f16, C3, resize_f16_c3);
impl_resize_image!(f16, C4, resize_f16_c4);
impl_resize_image!(f32, C1, resize_f32_c1);
impl_resize_image!(f32, C3, resize_f32_c3);
impl_resize_image!(f32, C4, resize_f32_c4);
impl_resize_image!(f32, AC4, resize_f32_ac4);

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: ResizeImage<T, L>,
{
    pub fn resize_into(
        stream_context: &StreamContext,
        resize: &Resize,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as ResizeImage<T, L>>::resize_image(stream_context, resize, source, destination)
    }

    pub fn resize_with(self, resize: Resize) -> Result<Self> {
        let mut destination = self
            .workspace
            .image::<T, L>(resize.destination_roi.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as ResizeImage<T, L>>::resize_image(
                self.stream_context,
                &resize,
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

    pub fn resize(self, size: Size, interpolation: InterpolationMode) -> Result<Self> {
        let source_size = self.size();
        let resize = Resize {
            source_roi: source_size.rectangle(),
            destination_roi: size.rectangle(),
            interpolation,
        };
        self.resize_with(resize)
    }
}
