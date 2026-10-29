use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ImagePipeline, operation_traits::SquareRootImage,
        operations_unary_methods::unary_power_methods::unary_power_image,
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SquareRootImage<T, L>,
{
    /// Computes the native square root of each processed sample.
    /// Integer implementations apply native fixed-point scaling with `scale_factor`;
    /// the provided floating-point implementations ignore `scale_factor`.
    /// Writes into the equally sized destination image.
    /// AC4 destination alpha retains its previous initialization state; this call does not initialize alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn square_root_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as SquareRootImage<T, L>>::square_root_image(
            stream_context,
            source,
            destination,
            scale_factor,
        )
    }

    /// Computes the native square root of each processed sample.
    /// Integer implementations apply native fixed-point scaling with `scale_factor`;
    /// the provided floating-point implementations ignore `scale_factor`.
    /// Updates the supplied image in place. AC4 alpha remains unchanged.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    pub fn square_root_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as SquareRootImage<T, L>>::square_root_image_in_place(
            stream_context,
            source_destination,
            scale_factor,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: SquareRootImage<T, L>,
{
    /// Computes the native square root of each processed sample.
    /// Integer implementations apply native fixed-point scaling with `scale_factor`;
    /// the provided floating-point implementations ignore `scale_factor`.
    /// Reuses owned image storage, or allocates owned output for borrowed input.
    /// For AC4, preserves the pipeline source alpha, copying it before out-of-place execution.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Allocating output, constructing views, or preserving source alpha can also fail.
    pub fn square_root(self, scale_factor: i32) -> Result<Self> {
        unary_power_image(
            self,
            scale_factor,
            <Self as SquareRootImage<T, L>>::square_root_image,
            <Self as SquareRootImage<T, L>>::square_root_image_in_place,
        )
    }
}
