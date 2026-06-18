use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, PlanarImageView, PlanarImageViewMut},
};

use super::super::{ImagePipeline, color::PlanarGammaImage};

impl<'a> ImagePipeline<'a, u8, C1>
where
    Self: PlanarGammaImage,
{
    pub fn gamma_forward_planar_into(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, u8, 3>,
        destination: &mut PlanarImageViewMut<'_, u8, 3>,
    ) -> Result<()> {
        <Self as PlanarGammaImage>::gamma_forward_planar_image(stream_context, source, destination)
    }

    pub fn gamma_forward_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, u8, 3>,
    ) -> Result<()> {
        <Self as PlanarGammaImage>::gamma_forward_planar_image_in_place(stream_context, image)
    }

    pub fn gamma_inverse_planar_into(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, u8, 3>,
        destination: &mut PlanarImageViewMut<'_, u8, 3>,
    ) -> Result<()> {
        <Self as PlanarGammaImage>::gamma_inverse_planar_image(stream_context, source, destination)
    }

    pub fn gamma_inverse_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, u8, 3>,
    ) -> Result<()> {
        <Self as PlanarGammaImage>::gamma_inverse_planar_image_in_place(stream_context, image)
    }
}
