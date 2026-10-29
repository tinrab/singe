use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView, ImageWindow},
    },
    pipeline::{
        ImageBacking,
        image::{
            ImagePipeline,
            filtering::{
                AdvancedGaussFilterImage, BilateralGaussBorderFilterImage, gaussian_geometry,
            },
        },
    },
    types::{BorderType, Point},
};

mod bilateral_gaussian_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: AdvancedGaussFilterImage<T, L>,
{
    /// Applies a separable Gaussian filter with caller-supplied floating-point coefficients.
    /// NPP expects an odd number of taps whose coefficients sum to one. Passes the kernel
    /// slice directly to NPP without checking coefficient values or normalization.
    /// The source window must retain a complete parent-image halo for the centered square
    /// mask derived from the kernel tap count.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Source-window and destination dimensions must match, and the parent halo must cover
    /// the complete neighborhood.
    /// The kernel must be nonempty and its tap count must fit `i32`. The wrapper also requires
    /// an odd tap count for the non-border operation.
    pub fn filter_gauss_advanced_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f32],
    ) -> Result<()> {
        <Self as AdvancedGaussFilterImage<T, L>>::filter_gauss_advanced_image(
            stream_context,
            source,
            destination,
            kernel,
        )
    }

    /// Applies a separable Gaussian filter with caller-supplied floating-point coefficients.
    /// NPP expects an odd number of taps whose coefficients sum to one. Passes the kernel
    /// slice directly to NPP without checking coefficient values or normalization.
    /// The native border routines support `BorderType::Replicate` and `BorderType::Mirror`;
    /// the wrapper forwards other values to NPP.
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions set its extent.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    /// The kernel must be nonempty and its tap count must fit `i32`. Native parameter failures
    /// are returned as errors.
    pub fn filter_gauss_advanced_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()> {
        <Self as AdvancedGaussFilterImage<T, L>>::filter_gauss_advanced_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            kernel,
            border_type,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: AdvancedGaussFilterImage<T, L>,
{
    /// Applies a separable Gaussian filter with caller-supplied floating-point coefficients.
    /// NPP expects an odd number of taps whose coefficients sum to one. Passes the kernel
    /// slice directly to NPP without checking coefficient values or normalization.
    /// Allocates a new owned image for the valid source window, reducing source dimensions
    /// by `kernel.len() - 1` in each dimension.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source must contain the complete mask and its valid output window.
    /// The kernel must be nonempty and its tap count must fit `i32`. The wrapper also requires
    /// an odd tap count for the non-border operation.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_gauss_advanced(self, kernel: &[f32]) -> Result<Self> {
        self.ensure_eager()?;
        let (mask, origin) = gaussian_geometry(kernel.len())?;
        let source = self.view()?.valid_window(mask, origin)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source.view())?;
            }
            <Self as AdvancedGaussFilterImage<T, L>>::filter_gauss_advanced_image(
                self.stream_context,
                &source,
                &mut destination_view,
                kernel,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    /// Applies a separable Gaussian filter with caller-supplied floating-point coefficients.
    /// NPP expects an odd number of taps whose coefficients sum to one. Passes the kernel
    /// slice directly to NPP without checking coefficient values or normalization.
    /// The native border routines support `BorderType::Replicate` and `BorderType::Mirror`;
    /// the wrapper forwards other values to NPP.
    /// `source_offset` locates the output ROI in the full source image. Allocates a new
    /// owned output from that offset to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    /// The kernel must be nonempty and its tap count must fit `i32`. Native parameter failures
    /// are returned as errors.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_gauss_advanced_border(
        self,
        source_offset: Point,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_source = source.window_from(source_offset)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), output_source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&output_source.view())?;
            }
            <Self as AdvancedGaussFilterImage<T, L>>::filter_gauss_advanced_border_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                kernel,
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
