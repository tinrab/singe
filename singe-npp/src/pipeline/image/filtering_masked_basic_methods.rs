use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageWindow},
    },
    pipeline::{
        ImageBacking,
        image::{
            ImagePipeline,
            filtering::{MaskedKernelFilterImage, centered_filter_geometry},
        },
    },
    types::MaskSize,
};

mod sharpen_methods;

mod into_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: MaskedKernelFilterImage<T, L>,
{
    /// Applies the native high-pass filter over the valid source window.
    /// Uses the selected square `mask_size` with a centered anchor. The wrapper rejects
    /// one-dimensional masks; NPP may reject unsupported square masks.
    /// Allocates a new owned image for the valid source window, reducing both dimensions
    /// by the selected square mask extent minus one.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source must contain the selected square mask and its valid output window.
    /// The mask must be square and supported by NPP.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_high_pass(self, mask_size: MaskSize) -> Result<Self> {
        self.masked_kernel_filter(
            mask_size,
            <Self as MaskedKernelFilterImage<T, L>>::filter_high_pass_image,
        )
    }

    /// Applies the native low-pass filter over the valid source window.
    /// Uses the selected square `mask_size` with a centered anchor. The wrapper rejects
    /// one-dimensional masks; NPP may reject unsupported square masks.
    /// Allocates a new owned image for the valid source window, reducing both dimensions
    /// by the selected square mask extent minus one.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source must contain the selected square mask and its valid output window.
    /// The mask must be square and supported by NPP.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_low_pass(self, mask_size: MaskSize) -> Result<Self> {
        self.masked_kernel_filter(
            mask_size,
            <Self as MaskedKernelFilterImage<T, L>>::filter_low_pass_image,
        )
    }

    /// Applies the native Gaussian filter over the valid source window.
    /// Uses the selected square `mask_size` with a centered anchor. The wrapper rejects
    /// one-dimensional masks; NPP may reject unsupported square masks.
    /// Allocates a new owned image for the valid source window, reducing both dimensions
    /// by the selected square mask extent minus one.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source must contain the selected square mask and its valid output window.
    /// The mask must be square and supported by NPP.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_gauss(self, mask_size: MaskSize) -> Result<Self> {
        self.masked_kernel_filter(
            mask_size,
            <Self as MaskedKernelFilterImage<T, L>>::filter_gauss_image,
        )
    }

    /// Applies the native Laplace filter over the valid source window.
    /// Uses the selected square `mask_size` with a centered anchor. The wrapper rejects
    /// one-dimensional masks; NPP may reject unsupported square masks.
    /// Allocates a new owned image for the valid source window, reducing both dimensions
    /// by the selected square mask extent minus one.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source must contain the selected square mask and its valid output window.
    /// The mask must be square and supported by NPP.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_laplace(self, mask_size: MaskSize) -> Result<Self> {
        self.masked_kernel_filter(
            mask_size,
            <Self as MaskedKernelFilterImage<T, L>>::filter_laplace_image,
        )
    }

    fn masked_kernel_filter(
        self,
        mask_size: MaskSize,
        filter: MaskedKernelFilterCallback<L, T>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let (mask, anchor) = centered_filter_geometry(mask_size)?;
        let source = self.view()?.valid_window(mask, anchor)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source.view())?;
            }
            filter(
                self.stream_context,
                &source,
                &mut destination_view,
                mask_size,
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

type MaskedKernelFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageWindow<'_, T, L>,
    &mut ImageOutputView<'_, T, L>,
    MaskSize,
) -> Result<()>;
