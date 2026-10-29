use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        operation_traits::{AlphaCompConstantImage, AlphaCompImage, AlphaPremultiplyConstantImage},
    },
    types::AlphaOperation,
};

mod constant_methods;
mod premultiply_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: AlphaCompImage<T, L>,
{
    /// Composites two images using their image alpha values and the selected alpha operation.
    ///
    /// The first image is `left` and the second is `right`. The provided
    /// implementations support native single-channel alpha composition and packed AC4
    /// composition. Alpha uses zero through the maximum channel value for integer samples,
    /// or `0.0..=1.0` for `f32`.
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
    pub fn alpha_comp_into(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        operation: AlphaOperation,
    ) -> Result<()> {
        <Self as AlphaCompImage<T, L>>::alpha_comp_image(
            stream_context,
            left,
            right,
            destination,
            operation,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: AlphaCompImage<T, L>,
{
    /// Composites two images using their image alpha values and the selected alpha operation.
    ///
    /// The first image is the pipeline image and the second is `other`. The provided
    /// implementations support native single-channel alpha composition and packed AC4
    /// composition. Alpha uses zero through the maximum channel value for integer samples,
    /// or `0.0..=1.0` for `f32`.
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
    pub fn alpha_comp(
        self,
        other: &ImageView<'_, T, L>,
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
            <Self as AlphaCompImage<T, L>>::alpha_comp_image(
                self.stream_context,
                &source,
                other,
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
