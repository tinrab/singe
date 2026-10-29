use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, PlanarImageOutputView, PlanarImageView, PlanarImageViewMut},
    pipeline::image::{ImagePipeline, color::PlanarGammaImage},
};

impl ImagePipeline<'_, u8, C1>
where
    Self: PlanarGammaImage,
{
    /// Applies native forward gamma correction to three separate `u8` image planes.
    /// Writes into the supplied destination planes.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, preparing regions or waiting for producers, CUDA
    /// context binding or validation, native gamma correction, or stream synchronization fails.
    /// Source and destination image dimensions must match.
    /// Native plane dimensions and byte strides must satisfy the planar gamma operation.
    pub fn gamma_forward_planar_into(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, u8, 3>,
        destination: &mut PlanarImageOutputView<'_, u8, 3>,
    ) -> Result<()> {
        <Self as PlanarGammaImage>::gamma_forward_planar_image(stream_context, source, destination)
    }

    /// Applies native forward gamma correction to three separate `u8` image planes.
    /// Updates the image in place.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, preparing regions or waiting for producers, CUDA
    /// context binding or validation, native gamma correction, or stream synchronization fails.
    /// Native plane dimensions and byte strides must satisfy the planar gamma operation.
    pub fn gamma_forward_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, u8, 3>,
    ) -> Result<()> {
        <Self as PlanarGammaImage>::gamma_forward_planar_image_in_place(stream_context, image)
    }

    /// Applies native inverse gamma correction to three separate `u8` image planes.
    /// Writes into the supplied destination planes.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, preparing regions or waiting for producers, CUDA
    /// context binding or validation, native gamma correction, or stream synchronization fails.
    /// Source and destination image dimensions must match.
    /// Native plane dimensions and byte strides must satisfy the planar gamma operation.
    pub fn gamma_inverse_planar_into(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, u8, 3>,
        destination: &mut PlanarImageOutputView<'_, u8, 3>,
    ) -> Result<()> {
        <Self as PlanarGammaImage>::gamma_inverse_planar_image(stream_context, source, destination)
    }

    /// Applies native inverse gamma correction to three separate `u8` image planes.
    /// Updates the image in place.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, preparing regions or waiting for producers, CUDA
    /// context binding or validation, native gamma correction, or stream synchronization fails.
    /// Native plane dimensions and byte strides must satisfy the planar gamma operation.
    pub fn gamma_inverse_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, u8, 3>,
    ) -> Result<()> {
        <Self as PlanarGammaImage>::gamma_inverse_planar_image_in_place(stream_context, image)
    }
}
