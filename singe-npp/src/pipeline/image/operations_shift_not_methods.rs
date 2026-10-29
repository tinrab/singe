use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageViewMut},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        operation_shapes::ConstantShiftImageOperation,
        operation_traits::{LeftShiftConstantImage, RightShiftConstantImage},
    },
};

mod logical_not_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: RightShiftConstantImage<T, L>,
    <Self as RightShiftConstantImage<T, L>>::Constant: Copy,
{
    /// Shifts each processed channel right by its constant bit count.
    ///
    /// For C1, `constant` is one `u32` count. C3, C4, and AC4 use arrays of three, four,
    /// and three counts respectively. Counts are passed as host values to NPP without
    /// wrapper validation of their range.
    /// Reuses an owned pipeline image in place; for borrowed input, allocates a new
    /// source-sized image.
    /// For AC4, preserves source alpha, copying the source before out-of-place execution
    /// when necessary.
    /// Row padding remains unwritten. Prepares participating regions, waits for producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Image view construction, output allocation, or copying source alpha can also fail.
    pub fn right_shift_constant(
        self,
        constant: <Self as RightShiftConstantImage<T, L>>::Constant,
    ) -> Result<Self> {
        shift_constant_image(
            self,
            constant,
            <Self as RightShiftConstantImage<T, L>>::right_shift_constant_image,
            <Self as RightShiftConstantImage<T, L>>::right_shift_constant_image_in_place,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: LeftShiftConstantImage<T, L>,
    <Self as LeftShiftConstantImage<T, L>>::Constant: Copy,
{
    /// Shifts each processed channel left by its constant bit count.
    ///
    /// For C1, `constant` is one `u32` count. C3, C4, and AC4 use arrays of three, four,
    /// and three counts respectively. Counts are passed as host values to NPP without
    /// wrapper validation of their range.
    /// Reuses an owned pipeline image in place; for borrowed input, allocates a new
    /// source-sized image.
    /// For AC4, preserves source alpha, copying the source before out-of-place execution
    /// when necessary.
    /// Row padding remains unwritten. Prepares participating regions, waits for producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Image view construction, output allocation, or copying source alpha can also fail.
    pub fn left_shift_constant(
        self,
        constant: <Self as LeftShiftConstantImage<T, L>>::Constant,
    ) -> Result<Self> {
        shift_constant_image(
            self,
            constant,
            <Self as LeftShiftConstantImage<T, L>>::left_shift_constant_image,
            <Self as LeftShiftConstantImage<T, L>>::left_shift_constant_image_in_place,
        )
    }
}

fn shift_constant_image<T, L, C>(
    mut pipeline: ImagePipeline<'_, T, L>,
    constant: C,
    operation: ConstantShiftImageOperation<T, L, C>,
    operation_in_place: for<'destination> fn(
        &StreamContext,
        C,
        &mut ImageViewMut<'destination, T, L>,
    ) -> Result<()>,
) -> Result<ImagePipeline<'_, T, L>>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    C: Copy,
{
    pipeline.ensure_eager()?;

    match &mut pipeline.backing {
        ImageBacking::Owned(image) => {
            let mut image_view = image.view_mut()?;
            operation_in_place(pipeline.stream_context, constant, &mut image_view)?;
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
                constant,
                &mut destination_view,
            )?;
            // SAFETY: the completed operation above wrote every element.
            pipeline.backing = ImageBacking::Owned(unsafe { destination.assume_init() });
        }
    }

    Ok(pipeline)
}
