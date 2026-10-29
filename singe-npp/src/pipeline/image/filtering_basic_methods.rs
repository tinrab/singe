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
            filtering::{BoxFilterImage, HighLowGaussFilterImage, centered_filter_geometry},
        },
    },
    types::{MaskSize, Point, Size},
};

mod into_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: HighLowGaussFilterImage<T, L>,
{
    /// Applies the native high-pass filter over the valid source window.
    /// Uses the selected square `mask_size` with its center as anchor. One-dimensional
    /// mask variants are rejected; NPP determines which square masks the routine supports.
    /// Allocates a new owned image for the valid source window, reducing source dimensions
    /// by the selected square mask extent minus one in each dimension.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source must contain the complete mask and its valid output window.
    /// The mask must be square and supported by NPP.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_high_pass_partial(self, mask_size: MaskSize) -> Result<Self> {
        self.high_low_gauss_filter(
            mask_size,
            <Self as HighLowGaussFilterImage<T, L>>::filter_high_pass_image,
        )
    }

    /// Applies the native low-pass filter over the valid source window.
    /// Uses the selected square `mask_size` with its center as anchor. One-dimensional
    /// mask variants are rejected; NPP determines which square masks the routine supports.
    /// Allocates a new owned image for the valid source window, reducing source dimensions
    /// by the selected square mask extent minus one in each dimension.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source must contain the complete mask and its valid output window.
    /// The mask must be square and supported by NPP.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_low_pass_partial(self, mask_size: MaskSize) -> Result<Self> {
        self.high_low_gauss_filter(
            mask_size,
            <Self as HighLowGaussFilterImage<T, L>>::filter_low_pass_image,
        )
    }

    /// Applies the native fixed-coefficient Gaussian filter over the valid source window.
    /// Uses the selected square `mask_size` with its center as anchor. One-dimensional
    /// mask variants are rejected; NPP determines which square masks the routine supports.
    /// Allocates a new owned image for the valid source window, reducing source dimensions
    /// by the selected square mask extent minus one in each dimension.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source must contain the complete mask and its valid output window.
    /// The mask must be square and supported by NPP.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_gauss_partial(self, mask_size: MaskSize) -> Result<Self> {
        self.high_low_gauss_filter(
            mask_size,
            <Self as HighLowGaussFilterImage<T, L>>::filter_gauss_image,
        )
    }

    fn high_low_gauss_filter(
        self,
        mask_size: MaskSize,
        filter: HighLowGaussFilterCallback<L, T>,
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

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: BoxFilterImage<T, L>,
{
    /// Applies the native box filter using `mask_size` and `anchor`.
    /// Allocates a new owned image for the valid source window, reducing source dimensions
    /// by `mask_size.width - 1` and `mask_size.height - 1`.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The source must contain the complete mask and its valid output window.
    /// Mask dimensions must be positive and the anchor must lie inside the mask.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_box(self, mask_size: Size, anchor: Point) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?.valid_window(mask_size, anchor)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source.view())?;
            }
            <Self as BoxFilterImage<T, L>>::filter_box_image(
                self.stream_context,
                &source,
                &mut destination_view,
                mask_size,
                anchor,
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

type HighLowGaussFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageWindow<'_, T, L>,
    &mut ImageOutputView<'_, T, L>,
    MaskSize,
) -> Result<()>;
