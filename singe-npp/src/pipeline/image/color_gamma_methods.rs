use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        color::{GammaImage, SameLayoutColorOperation},
    },
};

mod planar_methods;

impl<L> ImagePipeline<'_, u8, L>
where
    L: ChannelLayout,
    Self: GammaImage<L>,
{
    /// Applies native forward gamma correction to each active `u8` color channel.
    /// Writes into the supplied destination image.
    /// For AC4, destination alpha retains its previous initialization state; this call does not initialize alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, preparing regions or waiting for producers, CUDA
    /// context binding or validation, native gamma correction, or stream synchronization fails.
    /// Source and destination image dimensions must match.
    pub fn gamma_forward_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageOutputView<'_, u8, L>,
    ) -> Result<()> {
        <Self as GammaImage<L>>::gamma_forward_image(stream_context, source, destination)
    }

    /// Applies native forward gamma correction to each active `u8` color channel.
    /// Updates the image in place. Alpha is unchanged for AC4.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, preparing regions or waiting for producers, CUDA
    /// context binding or validation, native gamma correction, or stream synchronization fails.
    pub fn gamma_forward_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()> {
        <Self as GammaImage<L>>::gamma_forward_image_in_place(stream_context, image)
    }

    /// Applies native inverse gamma correction to each active `u8` color channel.
    /// Writes into the supplied destination image.
    /// For AC4, destination alpha retains its previous initialization state; this call does not initialize alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, preparing regions or waiting for producers, CUDA
    /// context binding or validation, native gamma correction, or stream synchronization fails.
    /// Source and destination image dimensions must match.
    pub fn gamma_inverse_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, u8, L>,
        destination: &mut ImageOutputView<'_, u8, L>,
    ) -> Result<()> {
        <Self as GammaImage<L>>::gamma_inverse_image(stream_context, source, destination)
    }

    /// Applies native inverse gamma correction to each active `u8` color channel.
    /// Updates the image in place. Alpha is unchanged for AC4.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, preparing regions or waiting for producers, CUDA
    /// context binding or validation, native gamma correction, or stream synchronization fails.
    pub fn gamma_inverse_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()> {
        <Self as GammaImage<L>>::gamma_inverse_image_in_place(stream_context, image)
    }
}

impl<L> ImagePipeline<'_, u8, L>
where
    L: ChannelLayout,
    u8: SupportedImage<L>,
    Self: GammaImage<L>,
{
    /// Applies native forward gamma correction to each active `u8` color channel.
    /// Reuses owned image storage, or allocates an owned image for borrowed input.
    /// For AC4, preserves source alpha, copying it before conversion when output is newly allocated.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, preparing regions or waiting for producers, CUDA
    /// context binding or validation, native gamma correction, or stream synchronization fails.
    /// Allocating output, constructing image views, or copying source alpha can also fail.
    pub fn gamma_forward(self) -> Result<Self> {
        self.gamma_with(
            <Self as GammaImage<L>>::gamma_forward_image,
            <Self as GammaImage<L>>::gamma_forward_image_in_place,
        )
    }

    /// Applies native inverse gamma correction to each active `u8` color channel.
    /// Reuses owned image storage, or allocates an owned image for borrowed input.
    /// For AC4, preserves source alpha, copying it before conversion when output is newly allocated.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, preparing regions or waiting for producers, CUDA
    /// context binding or validation, native gamma correction, or stream synchronization fails.
    /// Allocating output, constructing image views, or copying source alpha can also fail.
    pub fn gamma_inverse(self) -> Result<Self> {
        self.gamma_with(
            <Self as GammaImage<L>>::gamma_inverse_image,
            <Self as GammaImage<L>>::gamma_inverse_image_in_place,
        )
    }

    fn gamma_with(
        mut self,
        operation: SameLayoutColorOperation<L>,
        operation_in_place: for<'destination> fn(
            &StreamContext,
            &mut ImageViewMut<'destination, u8, L>,
        ) -> Result<()>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, &mut image_view)?;
            }
            ImageBacking::Borrowed(source) => {
                let mut destination = Image::<u8, L>::create_uninit(
                    self.stream_context.cuda_context(),
                    source.size(),
                )?;
                let mut destination_view = destination.output()?;
                if L::IGNORES_ALPHA {
                    destination_view.copy_from(source)?;
                }
                operation(self.stream_context, source, &mut destination_view)?;
                // SAFETY: completed gamma dispatch wrote active channels, and AC4 alpha was copied from input.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
