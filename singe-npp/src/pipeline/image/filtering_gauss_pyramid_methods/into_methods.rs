use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView},
    pipeline::image::{ImagePipeline, filtering::GaussPyramidBorderFilterImage},
    types::{BorderType, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: GaussPyramidBorderFilterImage<T, L>,
{
    /// Filters and downsamples a Gaussian pyramid layer with source border control.
    ///
    /// Uses separable `f32` kernel coefficients and a sampling `rate` greater than `1.0`
    /// and at most `10.0`. Noninteger rates use bilinear interpolation. The provided
    /// implementations accept C1 or C3 `u8`, `u16`, or `f32` images. `source_offset` locates
    /// the sampling origin in the full source image.
    /// The destination dimensions select the output extent.
    /// NPP expects kernel coefficients that sum to one and supports `BorderType::Mirror`
    /// or `BorderType::Replicate`. The wrapper checks that the kernel is nonempty and its
    /// length fits `i32`, then forwards coefficients and border selection without further
    /// validation. The slice address is passed directly without upload, context validation,
    /// or producer waits; coefficient storage must satisfy the native address-space
    /// requirements and remain ready and readable through native completion.
    /// Prepares source and destination regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native filtering, or stream synchronization fail.
    /// The source offset must address a source pixel, the rate must satisfy `1.0 < rate <= 10.0`,
    /// and the kernel must be nonempty with a length representable as `i32`. NPP may reject
    /// the supplied sampling geometry, coefficient array, or border type.
    pub fn filter_gauss_pyramid_layer_down_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()> {
        <Self as GaussPyramidBorderFilterImage<T, L>>::filter_gauss_pyramid_layer_down_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            rate,
            kernel,
            border_type,
        )
    }

    /// Filters and upsamples a Gaussian pyramid layer with source border control.
    ///
    /// Uses separable `f32` kernel coefficients and a sampling `rate` greater than `1.0`
    /// and at most `10.0`. Noninteger rates use bilinear interpolation. The provided
    /// implementations accept C1 or C3 `u8`, `u16`, or `f32` images. `source_offset` locates
    /// the sampling origin in the full source image.
    /// The destination dimensions select the output extent.
    /// NPP expects kernel coefficients that sum to one and supports `BorderType::Mirror`
    /// or `BorderType::Replicate`. The wrapper checks that the kernel is nonempty and its
    /// length fits `i32`, then forwards coefficients and border selection without further
    /// validation. The slice address is passed directly without upload, context validation,
    /// or producer waits; coefficient storage must satisfy the native address-space
    /// requirements and remain ready and readable through native completion.
    /// Prepares source and destination regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native filtering, or stream synchronization fail.
    /// The source offset must address a source pixel, the rate must satisfy `1.0 < rate <= 10.0`,
    /// and the kernel must be nonempty with a length representable as `i32`. NPP may reject
    /// the supplied sampling geometry, coefficient array, or border type.
    pub fn filter_gauss_pyramid_layer_up_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        rate: f32,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()> {
        <Self as GaussPyramidBorderFilterImage<T, L>>::filter_gauss_pyramid_layer_up_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            rate,
            kernel,
            border_type,
        )
    }
}
