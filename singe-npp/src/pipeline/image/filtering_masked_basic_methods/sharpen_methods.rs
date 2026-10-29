use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageWindow},
    },
    pipeline::image::{ImageBacking, ImagePipeline, filtering::DirectionalFilterImage},
    types::{Point, Size},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: DirectionalFilterImage<T, L>,
{
    /// Applies the native sharpening filter over the valid source window.
    /// Uses a centered 3-by-3 neighborhood.
    /// The source window must retain a complete parent-image halo for a centered 3-by-3
    /// neighborhood. Source-window and destination dimensions must match.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source window must match the destination size and retain the required halo.
    pub fn filter_sharpen_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()> {
        <Self as DirectionalFilterImage<T, L>>::filter_sharpen_image(
            stream_context,
            source,
            destination,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: DirectionalFilterImage<T, L>,
{
    /// Applies the native sharpening filter over the valid source window.
    /// Uses a centered 3-by-3 neighborhood.
    /// Allocates a new owned image for the valid source window, reducing both dimensions
    /// by two pixels.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source must contain the centered 3-by-3 neighborhood and its valid output window.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_sharpen(self) -> Result<Self> {
        self.ensure_eager()?;
        let source = self
            .view()?
            .valid_window(Size::new(3, 3), Point { x: 1, y: 1 })?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source.view())?;
            }
            <Self as DirectionalFilterImage<T, L>>::filter_sharpen_image(
                self.stream_context,
                &source,
                &mut destination_view,
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
