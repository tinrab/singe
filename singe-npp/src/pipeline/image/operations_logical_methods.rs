use crate::{
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageView},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        operation_shapes::{BinaryImageInPlaceOperation, BinaryImageOperation},
        operation_traits::LogicalBinaryImage,
    },
};

mod into_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: LogicalBinaryImage<T, L>,
{
    /// Applies bitwise AND to corresponding active image samples of both inputs.
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
    /// All participating images must have matching pixel dimensions.
    pub fn logical_and(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.logical_binary(
            other,
            <Self as LogicalBinaryImage<T, L>>::logical_and_image,
            <Self as LogicalBinaryImage<T, L>>::logical_and_image_in_place,
        )
    }

    /// Applies bitwise OR to corresponding active image samples of both inputs.
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
    /// All participating images must have matching pixel dimensions.
    pub fn logical_or(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.logical_binary(
            other,
            <Self as LogicalBinaryImage<T, L>>::logical_or_image,
            <Self as LogicalBinaryImage<T, L>>::logical_or_image_in_place,
        )
    }

    /// Applies bitwise XOR to corresponding active image samples of both inputs.
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
    /// All participating images must have matching pixel dimensions.
    pub fn logical_xor(self, other: &ImageView<'_, T, L>) -> Result<Self> {
        self.logical_binary(
            other,
            <Self as LogicalBinaryImage<T, L>>::logical_xor_image,
            <Self as LogicalBinaryImage<T, L>>::logical_xor_image_in_place,
        )
    }

    fn logical_binary(
        mut self,
        other: &ImageView<'_, T, L>,
        operation: BinaryImageOperation<T, L>,
        operation_in_place: BinaryImageInPlaceOperation<T, L>,
    ) -> Result<Self> {
        self.ensure_eager()?;

        match &mut self.backing {
            ImageBacking::Owned(image) => {
                let mut image_view = image.view_mut()?;
                operation_in_place(self.stream_context, other, &mut image_view)?;
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
                operation(self.stream_context, source, other, &mut destination_view)?;
                // SAFETY: the completed operation above wrote every element.
                self.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
            }
        }

        Ok(self)
    }
}
