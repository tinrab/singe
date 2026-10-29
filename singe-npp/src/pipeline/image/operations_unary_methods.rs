use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        operation_traits::{AbsoluteDifferenceImage, AbsoluteImage},
    },
};

mod unary_power_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: AbsoluteImage<T, L>,
{
    /// Computes the native absolute value of each processed sample.
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
    pub fn absolute_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as AbsoluteImage<T, L>>::absolute_image(stream_context, source, destination)
    }

    /// Computes the native absolute value of each processed sample.
    /// Updates the supplied image in place. AC4 alpha remains unchanged.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    pub fn absolute_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as AbsoluteImage<T, L>>::absolute_image_in_place(stream_context, source_destination)
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: AbsoluteImage<T, L>,
{
    /// Computes the native absolute value of each processed sample.
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
    pub fn absolute(mut self) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as AbsoluteImage<T, L>>::absolute_image_in_place(
                    self.stream_context,
                    &mut image_view,
                )?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<T, L>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                if L::IGNORES_ALPHA {
                    destination_view.copy_from(source)?;
                }
                if L::IGNORES_ALPHA {
                    destination_view.copy_from(source)?;
                }
                <Self as AbsoluteImage<T, L>>::absolute_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                )?;
                // SAFETY: completed dispatch wrote every active channel; any ignored alpha was copied from the source.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: AbsoluteDifferenceImage<T, L>,
{
    /// Computes the native absolute difference between corresponding image samples.
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
    /// Both input images must have matching pixel dimensions.
    pub fn absolute_difference_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as AbsoluteDifferenceImage<T, L>>::absolute_difference_image(
            stream_context,
            left,
            right,
            destination,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: AbsoluteDifferenceImage<T, L>,
{
    /// Computes the native absolute difference between corresponding image samples.
    /// Allocates a new owned output image even for owned input.
    /// For AC4, preserves the pipeline source alpha, copying it before out-of-place execution.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Allocating output, constructing views, or preserving source alpha can also fail.
    /// Both input images must have matching pixel dimensions.
    pub fn absolute_difference(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source)?;
            }
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source)?;
            }
            <Self as AbsoluteDifferenceImage<T, L>>::absolute_difference_image(
                self.stream_context,
                &source,
                other,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: completed dispatch wrote every active channel, and ignored alpha was preserved from the source.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
