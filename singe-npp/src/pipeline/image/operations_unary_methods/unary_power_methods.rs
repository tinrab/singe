use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline, operation_shapes::ScaledUnaryPowerImageOperation,
        operation_traits::SquareImage,
    },
};

mod square_root_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: SquareImage<T, L>,
{
    /// Squares each processed sample using the selected native arithmetic.
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
    pub fn square_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as SquareImage<T, L>>::square_image(stream_context, source, destination, scale_factor)
    }

    /// Squares each processed sample using the selected native arithmetic.
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
    pub fn square_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()> {
        <Self as SquareImage<T, L>>::square_image_in_place(
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
    Self: SquareImage<T, L>,
{
    /// Squares each processed sample using the selected native arithmetic.
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
    pub fn square(self, scale_factor: i32) -> Result<Self> {
        unary_power_image(
            self,
            scale_factor,
            <Self as SquareImage<T, L>>::square_image,
            <Self as SquareImage<T, L>>::square_image_in_place,
        )
    }
}

fn unary_power_image<T, L>(
    mut pipeline: ImagePipeline<'_, T, L>,
    scale_factor: i32,
    operation: ScaledUnaryPowerImageOperation<T, L>,
    operation_in_place: for<'destination> fn(
        &StreamContext,
        &mut ImageViewMut<'destination, T, L>,
        i32,
    ) -> Result<()>,
) -> Result<ImagePipeline<'_, T, L>>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
{
    pipeline.ensure_eager()?;

    match &mut pipeline.backing {
        ImageBacking::Owned(image) => {
            let mut image_view = image.view_mut()?;
            operation_in_place(pipeline.stream_context, &mut image_view, scale_factor)?;
        }
        ImageBacking::Borrowed(source) => {
            let mut destination = Image::<T, L>::create_uninit(
                pipeline.stream_context.cuda_context(),
                source.size(),
            )?;
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(source)?;
            }
            operation(
                pipeline.stream_context,
                source,
                &mut destination_view,
                scale_factor,
            )?;
            // SAFETY: the completed operation above wrote every element.
            pipeline.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
        }
    }

    Ok(pipeline)
}
