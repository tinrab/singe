use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        morphology::{ImageBacking, three_by_three_methods::ImagePipeline},
        morphology_traits::Morphology3x3BorderImage,
    },
    types::{BorderType, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Dilates each active channel using a centered 3-by-3 neighborhood.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// The destination dimensions select the output ROI extent.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for their producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel, checked ROI endpoints must fit inside
    /// the source, and output dimensions must be positive.
    pub fn dilate_3x3_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: Morphology3x3BorderImage<T, L>,
    {
        <Self as Morphology3x3BorderImage<T, L>>::dilate_3x3_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }

    /// Erodes each active channel using a centered 3-by-3 neighborhood.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// The destination dimensions select the output ROI extent.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for their producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel, checked ROI endpoints must fit inside
    /// the source, and output dimensions must be positive.
    pub fn erode_3x3_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()>
    where
        Self: Morphology3x3BorderImage<T, L>,
    {
        <Self as Morphology3x3BorderImage<T, L>>::erode_3x3_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: Morphology3x3BorderImage<T, L>,
{
    /// Dilates each active channel using a centered 3-by-3 neighborhood.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// Allocates a new owned image spanning from `source_offset` to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for their producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel, checked ROI endpoints must fit inside
    /// the source, and output dimensions must be positive.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn dilate_3x3_border(self, source_offset: Point, border_type: BorderType) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_window = source.window_from(source_offset)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), output_window.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&output_window.view())?;
            }
            <Self as Morphology3x3BorderImage<T, L>>::dilate_3x3_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                border_type,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    /// Erodes each active channel using a centered 3-by-3 neighborhood.
    /// `source_offset` locates the output ROI in the full source image.
    /// The native routines support `BorderType::Replicate`; other values are forwarded to NPP.
    /// Allocates a new owned image spanning from `source_offset` to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for their producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel, checked ROI endpoints must fit inside
    /// the source, and output dimensions must be positive.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn erode_3x3_border(self, source_offset: Point, border_type: BorderType) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_window = source.window_from(source_offset)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), output_window.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&output_window.view())?;
            }
            <Self as Morphology3x3BorderImage<T, L>>::erode_3x3_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                border_type,
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
