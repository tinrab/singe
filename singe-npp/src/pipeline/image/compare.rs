use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, ChannelLayout, ImageView, MaskOutputView},
    },
    pipeline::image::{ImageBacking, ImagePipeline, ImagePipeline as Pipeline},
    types::ComparisonOperation,
};

mod dispatch;

pub use dispatch::{
    CompareConstantImage, CompareEqualEpsilonConstantImage, CompareEqualEpsilonImage, CompareImage,
};

impl<T, L> Pipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CompareImage<T, L>,
{
    /// Compares corresponding pixels in two images using `operation`.
    /// Writes a C1 byte mask: `255` when every processed channel satisfies the comparison,
    /// and `0` otherwise. AC4 input alpha is excluded.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn compare_into(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        destination: &mut MaskOutputView<'_>,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as CompareImage<T, L>>::compare_image(
            stream_context,
            source1,
            source2,
            destination,
            operation,
        )
    }
}

impl<'a, T, L> Pipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    u8: SupportedImage<C1>,
    Self: CompareImage<T, L>,
{
    /// Compares corresponding pixels in two images using `operation`.
    /// Writes a C1 byte mask: `255` when every processed channel satisfies the comparison,
    /// and `0` otherwise. AC4 input alpha is excluded.
    /// Allocates a new owned C1 byte mask with the source dimensions.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    /// Allocation or image view construction can also fail.
    pub fn compare(
        self,
        other: &ImageView<'_, T, L>,
        operation: ComparisonOperation,
    ) -> Result<Pipeline<'a, u8, C1>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<u8, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as CompareImage<T, L>>::compare_image(
                self.stream_context,
                &source,
                other,
                &mut destination_view,
                operation,
            )?;
        }

        Ok(Pipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

impl<T, L> Pipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CompareConstantImage<T, L>,
{
    /// Compares source pixels with the supplied constant using `operation`.
    /// Writes a C1 byte mask: `255` when every processed channel satisfies the comparison,
    /// and `0` otherwise. AC4 input alpha is excluded.
    /// The provided implementations use a scalar constant for C1, three constants for C3
    /// and AC4, and four constants for C4.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn compare_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: <Self as CompareConstantImage<T, L>>::Constant,
        destination: &mut MaskOutputView<'_>,
        operation: ComparisonOperation,
    ) -> Result<()> {
        <Self as CompareConstantImage<T, L>>::compare_constant_image(
            stream_context,
            source,
            constant,
            destination,
            operation,
        )
    }
}

impl<'a, T, L> Pipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    u8: SupportedImage<C1>,
    Self: CompareConstantImage<T, L>,
{
    /// Compares source pixels with the supplied constant using `operation`.
    /// Writes a C1 byte mask: `255` when every processed channel satisfies the comparison,
    /// and `0` otherwise. AC4 input alpha is excluded.
    /// The provided implementations use a scalar constant for C1, three constants for C3
    /// and AC4, and four constants for C4.
    /// Allocates a new owned C1 byte mask with the source dimensions.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    /// Allocation or image view construction can also fail.
    pub fn compare_constant(
        self,
        constant: <Self as CompareConstantImage<T, L>>::Constant,
        operation: ComparisonOperation,
    ) -> Result<Pipeline<'a, u8, C1>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<u8, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as CompareConstantImage<T, L>>::compare_constant_image(
                self.stream_context,
                &source,
                constant,
                &mut destination_view,
                operation,
            )?;
        }

        Ok(Pipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}

mod epsilon_methods;
