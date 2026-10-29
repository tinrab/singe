use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        operation_shapes::{ConstantInPlaceOperation, ConstantOperation},
        operation_traits::LogicalConstantImage,
    },
};

mod xor_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: LogicalConstantImage<T, L>,
    <Self as LogicalConstantImage<T, L>>::Constant: Copy,
{
    /// Applies bitwise AND to corresponding active image samples and `constant`.
    /// The provided single-channel implementation uses a scalar constant; packed layouts use
    /// one constant per processed channel, including three constants for AC4.
    /// Writes into `destination`. AC4 alpha retains its previous initialization state; this
    /// operation does not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination dimensions must match.
    pub fn logical_and_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalConstantImage<T, L>>::logical_and_constant_image(
            stream_context,
            source,
            constant,
            destination,
        )
    }

    /// Applies bitwise AND to corresponding active image samples and `constant`.
    /// The provided single-channel implementation uses a scalar constant; packed layouts use
    /// one constant per processed channel, including three constants for AC4.
    /// Updates `source_destination` in place. AC4 alpha remains unchanged.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    pub fn logical_and_constant_in_place(
        stream_context: &StreamContext,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalConstantImage<T, L>>::logical_and_constant_image_in_place(
            stream_context,
            constant,
            source_destination,
        )
    }

    /// Applies bitwise OR to corresponding active image samples and `constant`.
    /// The provided single-channel implementation uses a scalar constant; packed layouts use
    /// one constant per processed channel, including three constants for AC4.
    /// Writes into `destination`. AC4 alpha retains its previous initialization state; this
    /// operation does not initialize that channel.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source and destination dimensions must match.
    pub fn logical_or_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalConstantImage<T, L>>::logical_or_constant_image(
            stream_context,
            source,
            constant,
            destination,
        )
    }

    /// Applies bitwise OR to corresponding active image samples and `constant`.
    /// The provided single-channel implementation uses a scalar constant; packed layouts use
    /// one constant per processed channel, including three constants for AC4.
    /// Updates `source_destination` in place. AC4 alpha remains unchanged.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    pub fn logical_or_constant_in_place(
        stream_context: &StreamContext,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()> {
        <Self as LogicalConstantImage<T, L>>::logical_or_constant_image_in_place(
            stream_context,
            constant,
            source_destination,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: LogicalConstantImage<T, L>,
    <Self as LogicalConstantImage<T, L>>::Constant: Copy,
{
    /// Applies bitwise AND to corresponding active image samples and `constant`.
    /// The provided single-channel implementation uses a scalar constant; packed layouts use
    /// one constant per processed channel, including three constants for AC4.
    /// Reuses owned image storage, or allocates an owned image for borrowed input.
    /// For AC4, preserves the pipeline source alpha, copying it before an out-of-place operation.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Constructing image views, allocating output, or preserving source alpha can also fail.
    pub fn logical_and_constant(
        self,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
    ) -> Result<Self> {
        self.logical_constant(
            constant,
            <Self as LogicalConstantImage<T, L>>::logical_and_constant_image,
            <Self as LogicalConstantImage<T, L>>::logical_and_constant_image_in_place,
        )
    }

    /// Applies bitwise OR to corresponding active image samples and `constant`.
    /// The provided single-channel implementation uses a scalar constant; packed layouts use
    /// one constant per processed channel, including three constants for AC4.
    /// Reuses owned image storage, or allocates an owned image for borrowed input.
    /// For AC4, preserves the pipeline source alpha, copying it before an out-of-place operation.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Constructing image views, allocating output, or preserving source alpha can also fail.
    pub fn logical_or_constant(
        self,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
    ) -> Result<Self> {
        self.logical_constant(
            constant,
            <Self as LogicalConstantImage<T, L>>::logical_or_constant_image,
            <Self as LogicalConstantImage<T, L>>::logical_or_constant_image_in_place,
        )
    }

    pub(crate) fn logical_constant(
        mut self,
        constant: <Self as LogicalConstantImage<T, L>>::Constant,
        operation: ConstantOperation<T, L, <Self as LogicalConstantImage<T, L>>::Constant>,
        operation_in_place: ConstantInPlaceOperation<
            T,
            L,
            <Self as LogicalConstantImage<T, L>>::Constant,
        >,
    ) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, constant, &mut image_view)?;
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
                operation(self.stream_context, source, constant, &mut destination_view)?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
