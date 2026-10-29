use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::{
        ImageBacking,
        image::{ImagePipeline, filtering::MaskedKernelBorderFilterImage},
    },
    types::{BorderType, MaskSize, Point},
};

mod into_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: MaskedKernelBorderFilterImage<T, L>,
{
    /// Applies the native high-pass filter with border control.
    /// `mask_size` selects the native filter kernel. The wrapper forwards mask and border
    /// values to NPP, which may reject unsupported combinations.
    /// `source_offset` locates the output ROI in the full source. Allocates a new owned
    /// output spanning from that offset to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_high_pass_border(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.masked_kernel_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as MaskedKernelBorderFilterImage<T, L>>::filter_high_pass_border_image,
        )
    }

    /// Applies the native low-pass filter with border control.
    /// `mask_size` selects the native filter kernel. The wrapper forwards mask and border
    /// values to NPP, which may reject unsupported combinations.
    /// `source_offset` locates the output ROI in the full source. Allocates a new owned
    /// output spanning from that offset to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_low_pass_border(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.masked_kernel_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as MaskedKernelBorderFilterImage<T, L>>::filter_low_pass_border_image,
        )
    }

    /// Applies the native Gaussian filter with border control.
    /// `mask_size` selects the native filter kernel. The wrapper forwards mask and border
    /// values to NPP, which may reject unsupported combinations.
    /// `source_offset` locates the output ROI in the full source. Allocates a new owned
    /// output spanning from that offset to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_gauss_border(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.masked_kernel_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as MaskedKernelBorderFilterImage<T, L>>::filter_gauss_border_image,
        )
    }

    /// Applies the native Laplace filter with border control.
    /// `mask_size` selects the native filter kernel. The wrapper forwards mask and border
    /// values to NPP, which may reject unsupported combinations.
    /// `source_offset` locates the output ROI in the full source. Allocates a new owned
    /// output spanning from that offset to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_laplace_border(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.masked_kernel_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as MaskedKernelBorderFilterImage<T, L>>::filter_laplace_border_image,
        )
    }

    fn masked_kernel_border_filter(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
        filter: MaskedKernelBorderFilterCallback<L, T>,
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
            filter(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                mask_size,
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

type MaskedKernelBorderFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    Point,
    &mut ImageOutputView<'_, T, L>,
    MaskSize,
    BorderType,
) -> Result<()>;
