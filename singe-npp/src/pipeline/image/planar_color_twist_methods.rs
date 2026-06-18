use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        view::{C2, PlanarImageView, PlanarImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
};

use super::{PlanarImage, SemiplanarImage};

type PlanarColorTwist<T> = for<'source, 'destination> fn(
    &StreamContext,
    &PlanarImageView<'source, T, 3>,
    &mut PlanarImageViewMut<'destination, T, 3>,
    color::ColorTwistMatrix,
) -> Result<()>;

impl PlanarImage<f32, 3> {
    pub fn color_twist(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<Self> {
        self.color_twist_with(stream_context, twist, color::color_twist_f32_p3)
    }

    fn color_twist_with(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
        operation: PlanarColorTwist<f32>,
    ) -> Result<Self> {
        let mut destination = Self::create(self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(stream_context, &source, &mut destination_view, twist)?;
        }

        Ok(destination)
    }
}

impl PlanarImage<u16, 3> {
    pub fn color_twist(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<Self> {
        self.color_twist_with(stream_context, twist, color::color_twist_u16_p3)
    }

    pub fn rgb_to_nv12(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<SemiplanarImage<u16>>
    where
        Workspace: ImageAllocator<u16, C2>,
    {
        self.color_twist_to_nv12(stream_context, twist, color::rgb_to_nv12_u16_color_twist_p3)
    }

    fn color_twist_with(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
        operation: PlanarColorTwist<u16>,
    ) -> Result<Self> {
        let mut destination = Self::create(self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(stream_context, &source, &mut destination_view, twist)?;
        }

        Ok(destination)
    }
}

impl PlanarImage<i8, 3> {
    pub fn color_twist_into(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, i8, 3>,
        destination: &mut PlanarImageViewMut<'_, i8, 3>,
        twist: color::ColorTwistMatrix,
    ) -> Result<()> {
        color::color_twist_i8_p3(stream_context, source, destination, twist)
    }
}

impl PlanarImage<i16, 3> {
    pub fn color_twist(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<Self> {
        self.color_twist_with(stream_context, twist, color::color_twist_i16_p3)
    }

    fn color_twist_with(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
        operation: PlanarColorTwist<i16>,
    ) -> Result<Self> {
        let mut destination = Self::create(self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(stream_context, &source, &mut destination_view, twist)?;
        }

        Ok(destination)
    }
}
