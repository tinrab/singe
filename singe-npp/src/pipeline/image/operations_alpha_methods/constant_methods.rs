use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking,
        operations_alpha_methods::{AlphaCompConstantImage, ImagePipeline},
    },
    types::AlphaOperation,
};

mod premultiply_constant_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: AlphaCompConstantImage<T, L>,
    <Self as AlphaCompConstantImage<T, L>>::Alpha: Copy,
{
    /// Composites two images using constant opacities and the selected alpha operation.
    ///
    /// `alpha` applies to the first source and `other_alpha` to `other`. Opacity uses
    /// zero through the maximum channel value for integer samples, or `0.0..=1.0` for
    /// `f32`. The wrapper forwards opacity values without range validation.
    /// Forwards `operation` to NPP without a wrapper whitelist. All participating image
    /// dimensions must match.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful native execution and stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native alpha processing, or stream synchronization fail.
    /// Participating image dimensions must match. NPP may reject the supplied alpha
    /// operation or opacity parameters.
    pub fn alpha_comp_constant_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        alpha: <Self as AlphaCompConstantImage<T, L>>::Alpha,
        other: &ImageView<'_, T, L>,
        other_alpha: <Self as AlphaCompConstantImage<T, L>>::Alpha,
        destination: &mut ImageOutputView<'_, T, L>,
        operation: AlphaOperation,
    ) -> Result<()> {
        <Self as AlphaCompConstantImage<T, L>>::alpha_comp_constant_image(
            stream_context,
            source,
            alpha,
            other,
            other_alpha,
            destination,
            operation,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: AlphaCompConstantImage<T, L>,
    <Self as AlphaCompConstantImage<T, L>>::Alpha: Copy,
{
    /// Composites two images using constant opacities and the selected alpha operation.
    ///
    /// `alpha` applies to the first source and `other_alpha` to `other`. Opacity uses
    /// zero through the maximum channel value for integer samples, or `0.0..=1.0` for
    /// `f32`. The wrapper forwards opacity values without range validation.
    /// Forwards `operation` to NPP without a wrapper whitelist. All participating image
    /// dimensions must match.
    /// Always allocates a new source-sized image. For AC4, copies the pipeline source into
    /// the destination before native composition.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful native execution and stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native alpha processing, or stream synchronization fail.
    /// Participating image dimensions must match. NPP may reject the supplied alpha
    /// operation or opacity parameters.
    /// Output allocation, image view construction, or copying the source can also fail.
    pub fn alpha_comp_constant(
        self,
        alpha: <Self as AlphaCompConstantImage<T, L>>::Alpha,
        other: &ImageView<'_, T, L>,
        other_alpha: <Self as AlphaCompConstantImage<T, L>>::Alpha,
        operation: AlphaOperation,
    ) -> Result<Self> {
        self.ensure_eager()?;

        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source)?;
            }
            <Self as AlphaCompConstantImage<T, L>>::alpha_comp_constant_image(
                self.stream_context,
                &source,
                alpha,
                other,
                other_alpha,
                &mut destination_view,
                operation,
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
