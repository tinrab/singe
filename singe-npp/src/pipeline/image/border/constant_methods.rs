use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking,
        border::{ConstantBorderImage, ImagePipeline},
    },
    types::Size,
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ConstantBorderImage<T, L>,
    <Self as ConstantBorderImage<T, L>>::Value: Copy,
{
    /// Copies the source into a destination and fills the surrounding border with `value`.
    /// `top` and `left` are pixel offsets of the source origin in the destination. The source
    /// rectangle must fit completely; the remaining right and bottom margins form the border.
    /// The provided implementations use a scalar value for C1 and one value per physical
    /// channel for packed images, including four values for AC4.
    /// Processes every physical channel, including alpha for AC4.
    /// Writes every logical sample of the supplied destination image.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Offsets must fit `i32`, placement endpoints must not overflow, and the source placement
    /// must fit within the destination. NPP may reject additional native geometry restrictions.
    pub fn constant_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        top: usize,
        left: usize,
        value: <Self as ConstantBorderImage<T, L>>::Value,
    ) -> Result<()> {
        <Self as ConstantBorderImage<T, L>>::copy_constant_border_image(
            stream_context,
            source,
            destination,
            top,
            left,
            value,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: ConstantBorderImage<T, L>,
    <Self as ConstantBorderImage<T, L>>::Value: Copy,
{
    /// Copies the source into a destination and fills the surrounding border with `value`.
    /// `top` and `left` are pixel offsets of the source origin in the destination. The source
    /// rectangle must fit completely; the remaining right and bottom margins form the border.
    /// The provided implementations use a scalar value for C1 and one value per physical
    /// channel for packed images, including four values for AC4.
    /// Processes every physical channel, including alpha for AC4.
    /// Allocates a new owned image with the requested `size`.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Offsets must fit `i32`, placement endpoints must not overflow, and the source placement
    /// must fit within the destination. NPP may reject additional native geometry restrictions.
    /// Output allocation, image-view construction, or copying existing pixels can also fail.
    pub fn constant_border(
        self,
        size: Size,
        top: usize,
        left: usize,
        value: <Self as ConstantBorderImage<T, L>>::Value,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), size)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as ConstantBorderImage<T, L>>::copy_constant_border_image(
                self.stream_context,
                &source,
                &mut destination_view,
                top,
                left,
                value,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
