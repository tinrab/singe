use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{
            C1, ChannelLayout, ImageOutputView, ImageView, ImageViewMut, PlanarImageOutputView,
            PlanarImageView, PlanarImageViewMut,
        },
    },
    pipeline::image::{
        ColorTwistMatrix, ImageBacking, ImagePipeline,
        color::{ColorTwistImage, PlanarColorTwistImage},
    },
};

mod batch_methods;
mod constants_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ColorTwistImage<T, L>,
{
    /// Applies the native color transform specified by the three-row, four-column `twist` matrix.
    /// Matrix coefficients are forwarded without numerical validation.
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
    pub fn color_twist_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        twist: ColorTwistMatrix,
    ) -> Result<()> {
        <Self as ColorTwistImage<T, L>>::color_twist_image(
            stream_context,
            source,
            destination,
            twist,
        )
    }

    /// Applies the native color transform specified by the three-row, four-column `twist` matrix.
    /// Matrix coefficients are forwarded without numerical validation.
    /// Updates the supplied image in place. AC4 alpha remains unchanged.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    pub fn color_twist_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, T, L>,
        twist: ColorTwistMatrix,
    ) -> Result<()> {
        <Self as ColorTwistImage<T, L>>::color_twist_image_in_place(stream_context, image, twist)
    }
}

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
    Self: PlanarColorTwistImage<T>,
{
    /// Applies the native color transform specified by the three-row, four-column `twist` matrix.
    /// Matrix coefficients are forwarded without numerical validation.
    /// Processes three separate image planes using the native plane ordering and common-stride
    /// requirements.
    /// Writes into the equally sized destination planes.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn color_twist_planar_into(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, T, 3>,
        destination: &mut PlanarImageOutputView<'_, T, 3>,
        twist: ColorTwistMatrix,
    ) -> Result<()> {
        <Self as PlanarColorTwistImage<T>>::color_twist_planar_image(
            stream_context,
            source,
            destination,
            twist,
        )
    }

    /// Applies the native color transform specified by the three-row, four-column `twist` matrix.
    /// Matrix coefficients are forwarded without numerical validation.
    /// Processes three separate image planes using the native plane ordering and common-stride
    /// requirements.
    /// Updates the three supplied planes in place.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    pub fn color_twist_planar_in_place(
        stream_context: &StreamContext,
        image: &mut PlanarImageViewMut<'_, T, 3>,
        twist: ColorTwistMatrix,
    ) -> Result<()> {
        <Self as PlanarColorTwistImage<T>>::color_twist_planar_image_in_place(
            stream_context,
            image,
            twist,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: ColorTwistImage<T, L>,
{
    /// Applies the native color transform specified by the three-row, four-column `twist` matrix.
    /// Matrix coefficients are forwarded without numerical validation.
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
    pub fn color_twist(mut self, twist: ColorTwistMatrix) -> Result<Self> {
        self.ensure_eager()?;
        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                <Self as ColorTwistImage<T, L>>::color_twist_image_in_place(
                    self.stream_context,
                    &mut image_view,
                    twist,
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
                <Self as ColorTwistImage<T, L>>::color_twist_image(
                    self.stream_context,
                    source,
                    &mut destination_view,
                    twist,
                )?;
                // SAFETY: completed pointwise dispatch initialized active channels, and ignored alpha was preserved.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
