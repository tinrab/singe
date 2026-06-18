use crate::{
    context::StreamContext,
    error::Result,
    image::{
        memory::Image,
        view::{AC4, C3, PlanarImageView, PlanarImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::PlanarImage;

type Planar4ToPackedColorConvert = for<'source, 'destination> fn(
    &StreamContext,
    &PlanarImageView<'source, u8, 4>,
    &mut crate::image::view::ImageViewMut<'destination, u8, AC4>,
) -> Result<()>;

type Planar4ToC3ColorConvert = for<'source, 'destination> fn(
    &StreamContext,
    &PlanarImageView<'source, u8, 4>,
    &mut crate::image::view::ImageViewMut<'destination, u8, C3>,
) -> Result<()>;

type Planar4ToPlanar4ColorConvert = for<'source, 'destination> fn(
    &StreamContext,
    &PlanarImageView<'source, u8, 4>,
    &mut PlanarImageViewMut<'destination, u8, 4>,
) -> Result<()>;

type Planar4ToPlanar3ColorConvert = for<'source, 'destination> fn(
    &StreamContext,
    &PlanarImageView<'source, u8, 4>,
    &mut PlanarImageViewMut<'destination, u8, 3>,
) -> Result<()>;

pub(super) trait PlanarP4ColorConversionExt {
    fn color_convert(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToPlanar4ColorConvert,
    ) -> Result<Self>
    where
        Self: Sized;

    fn color_convert_to_ac4(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToPackedColorConvert,
    ) -> Result<Image<u8, AC4>>
    where
        Workspace: ImageAllocator<u8, AC4>;

    fn color_convert_to_p3(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToPlanar3ColorConvert,
    ) -> Result<PlanarImage<u8, 3>>;

    fn color_convert_to_c3(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToC3ColorConvert,
    ) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>;
}

impl PlanarP4ColorConversionExt for PlanarImage<u8, 4> {
    fn color_convert(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToPlanar4ColorConvert,
    ) -> Result<Self> {
        let mut destination = Self::create(self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(stream_context, &source, &mut destination_view)?;
        }

        Ok(destination)
    }

    fn color_convert_to_ac4(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToPackedColorConvert,
    ) -> Result<Image<u8, AC4>>
    where
        Workspace: ImageAllocator<u8, AC4>,
    {
        let mut destination = Workspace::create_image(self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(stream_context, &source, &mut destination_view)?;
        }

        Ok(destination)
    }

    fn color_convert_to_p3(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToPlanar3ColorConvert,
    ) -> Result<PlanarImage<u8, 3>> {
        let mut destination = PlanarImage::<u8, 3>::create(self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(stream_context, &source, &mut destination_view)?;
        }

        Ok(destination)
    }

    fn color_convert_to_c3(
        self,
        stream_context: &StreamContext,
        operation: Planar4ToC3ColorConvert,
    ) -> Result<Image<u8, C3>>
    where
        Workspace: ImageAllocator<u8, C3>,
    {
        let mut destination = Workspace::create_image(self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(stream_context, &source, &mut destination_view)?;
        }

        Ok(destination)
    }
}
