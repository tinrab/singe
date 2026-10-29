use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{C1, ChannelLayout, ImageView, MaskOutputView},
    },
    pipeline::image::{
        ImageBacking,
        compare::{
            CompareEqualEpsilonConstantImage, CompareEqualEpsilonImage, ImagePipeline as Pipeline,
        },
    },
};

impl<L> Pipeline<'_, f32, L>
where
    L: ChannelLayout,
    Self: CompareEqualEpsilonImage<L>,
{
    /// Compares corresponding pixels in two images for equality within `epsilon`.
    /// Writes a C1 byte mask: `255` when every processed channel satisfies the comparison,
    /// and `0` otherwise. AC4 input alpha is excluded.
    /// `epsilon` is the tolerance applied to absolute sample differences and is forwarded
    /// to NPP without validation.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn compare_equal_epsilon_into(
        stream_context: &StreamContext,
        source1: &ImageView<'_, f32, L>,
        source2: &ImageView<'_, f32, L>,
        destination: &mut MaskOutputView<'_>,
        epsilon: f32,
    ) -> Result<()> {
        <Self as CompareEqualEpsilonImage<L>>::compare_equal_epsilon_image(
            stream_context,
            source1,
            source2,
            destination,
            epsilon,
        )
    }
}

impl<'a, L> Pipeline<'a, f32, L>
where
    L: ChannelLayout,
    u8: SupportedImage<C1>,
    Self: CompareEqualEpsilonImage<L>,
{
    /// Compares corresponding pixels in two images for equality within `epsilon`.
    /// Writes a C1 byte mask: `255` when every processed channel satisfies the comparison,
    /// and `0` otherwise. AC4 input alpha is excluded.
    /// `epsilon` is the tolerance applied to absolute sample differences and is forwarded
    /// to NPP without validation.
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
    pub fn compare_equal_epsilon(
        self,
        other: &ImageView<'_, f32, L>,
        epsilon: f32,
    ) -> Result<Pipeline<'a, u8, C1>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<u8, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as CompareEqualEpsilonImage<L>>::compare_equal_epsilon_image(
                self.stream_context,
                &source,
                other,
                &mut destination_view,
                epsilon,
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

impl<L> Pipeline<'_, f32, L>
where
    L: ChannelLayout,
    Self: CompareEqualEpsilonConstantImage<L>,
{
    /// Compares source pixels with the supplied constant for equality within `epsilon`.
    /// Writes a C1 byte mask: `255` when every processed channel satisfies the comparison,
    /// and `0` otherwise. AC4 input alpha is excluded.
    /// The provided implementations use a scalar constant for C1, three constants for C3
    /// and AC4, and four constants for C4.
    /// `epsilon` is the tolerance applied to absolute sample differences and is forwarded
    /// to NPP without validation.
    /// Writes every logical sample of the supplied destination.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// All participating image dimensions must match.
    pub fn compare_equal_epsilon_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, f32, L>,
        constant: <Self as CompareEqualEpsilonConstantImage<L>>::Constant,
        destination: &mut MaskOutputView<'_>,
        epsilon: f32,
    ) -> Result<()> {
        <Self as CompareEqualEpsilonConstantImage<L>>::compare_equal_epsilon_constant_image(
            stream_context,
            source,
            constant,
            destination,
            epsilon,
        )
    }
}

impl<'a, L> Pipeline<'a, f32, L>
where
    L: ChannelLayout,
    u8: SupportedImage<C1>,
    Self: CompareEqualEpsilonConstantImage<L>,
{
    /// Compares source pixels with the supplied constant for equality within `epsilon`.
    /// Writes a C1 byte mask: `255` when every processed channel satisfies the comparison,
    /// and `0` otherwise. AC4 input alpha is excluded.
    /// The provided implementations use a scalar constant for C1, three constants for C3
    /// and AC4, and four constants for C4.
    /// `epsilon` is the tolerance applied to absolute sample differences and is forwarded
    /// to NPP without validation.
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
    pub fn compare_equal_epsilon_constant(
        self,
        constant: <Self as CompareEqualEpsilonConstantImage<L>>::Constant,
        epsilon: f32,
    ) -> Result<Pipeline<'a, u8, C1>> {
        self.ensure_eager()?;

        let mut destination =
            Image::<u8, C1>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            <Self as CompareEqualEpsilonConstantImage<L>>::compare_equal_epsilon_constant_image(
                self.stream_context,
                &source,
                constant,
                &mut destination_view,
                epsilon,
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
