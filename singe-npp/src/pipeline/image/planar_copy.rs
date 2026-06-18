use crate::{
    context::StreamContext,
    error::Result,
    image::{
        exchange,
        view::{
            C1, C3, C4, ChannelLayout, ImageView, ImageViewMut, PlanarImageView, PlanarImageViewMut,
        },
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::planar::{CreatePlanarImage, PlanarImage};
use super::{ImageBacking, ImagePipeline};

pub trait PackedToPlanarImage<T, L, const C: usize> {
    fn copy_packed_to_planar_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut PlanarImageViewMut<'_, T, C>,
    ) -> Result<()>;
}

pub trait PlanarToPackedImage<T, L, const C: usize> {
    fn copy_planar_to_packed_image(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, T, C>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

macro_rules! impl_packed_to_planar_image {
    ($ty:ty, $layout:ty, $planes:literal, $copy:path) => {
        impl<'a> PackedToPlanarImage<$ty, $layout, $planes> for ImagePipeline<'a, $ty, $layout> {
            fn copy_packed_to_planar_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut PlanarImageViewMut<'_, $ty, $planes>,
            ) -> Result<()> {
                $copy(stream_context, source, destination)
            }
        }
    };
}

macro_rules! impl_planar_to_packed_image {
    ($ty:ty, $layout:ty, $planes:literal, $copy:path) => {
        impl<'a> PlanarToPackedImage<$ty, $layout, $planes> for ImagePipeline<'a, $ty, $layout> {
            fn copy_planar_to_packed_image(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, $ty, $planes>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $copy(stream_context, source, destination)
            }
        }
    };
}

impl_packed_to_planar_image!(u8, C3, 3, exchange::copy_c3_to_p3);
impl_packed_to_planar_image!(u8, C4, 4, exchange::copy_c4_to_p4);
impl_packed_to_planar_image!(u16, C3, 3, exchange::copy_c3_to_p3);
impl_packed_to_planar_image!(u16, C4, 4, exchange::copy_c4_to_p4);
impl_packed_to_planar_image!(i16, C3, 3, exchange::copy_c3_to_p3);
impl_packed_to_planar_image!(i16, C4, 4, exchange::copy_c4_to_p4);
impl_packed_to_planar_image!(i32, C3, 3, exchange::copy_c3_to_p3);
impl_packed_to_planar_image!(i32, C4, 4, exchange::copy_c4_to_p4);
impl_packed_to_planar_image!(f32, C3, 3, exchange::copy_c3_to_p3);
impl_packed_to_planar_image!(f32, C4, 4, exchange::copy_c4_to_p4);

impl_planar_to_packed_image!(u8, C3, 3, exchange::copy_p3_to_c3);
impl_planar_to_packed_image!(u8, C4, 4, exchange::copy_p4_to_c4);
impl_planar_to_packed_image!(u16, C3, 3, exchange::copy_p3_to_c3);
impl_planar_to_packed_image!(u16, C4, 4, exchange::copy_p4_to_c4);
impl_planar_to_packed_image!(i16, C3, 3, exchange::copy_p3_to_c3);
impl_planar_to_packed_image!(i16, C4, 4, exchange::copy_p4_to_c4);
impl_planar_to_packed_image!(i32, C3, 3, exchange::copy_p3_to_c3);
impl_planar_to_packed_image!(i32, C4, 4, exchange::copy_p4_to_c4);
impl_planar_to_packed_image!(f32, C3, 3, exchange::copy_p3_to_c3);
impl_planar_to_packed_image!(f32, C4, 4, exchange::copy_p4_to_c4);

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn copy_to_planar<const C: usize>(self) -> Result<PlanarImage<T, C>>
    where
        Workspace: ImageAllocator<T, C1>,
        PlanarImage<T, C>: CreatePlanarImage<T, C>,
        Self: PackedToPlanarImage<T, L, C>,
    {
        let mut destination = <PlanarImage<T, C> as CreatePlanarImage<T, C>>::create(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view =
                <PlanarImage<T, C> as CreatePlanarImage<T, C>>::view_mut(&mut destination)?;
            <Self as PackedToPlanarImage<T, L, C>>::copy_packed_to_planar_image(
                self.stream_context,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(destination)
    }

    pub fn copy_from_planar<const C: usize>(
        self,
        source: &PlanarImageView<'_, T, C>,
    ) -> Result<Self>
    where
        Workspace: ImageAllocator<T, L>,
        Self: PlanarToPackedImage<T, L, C>,
    {
        let mut destination = self.workspace.image::<T, L>(self.size())?;

        {
            let mut destination_view = destination.view_mut()?;
            <Self as PlanarToPackedImage<T, L, C>>::copy_planar_to_packed_image(
                self.stream_context,
                source,
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
