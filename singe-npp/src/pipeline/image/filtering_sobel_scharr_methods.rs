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
            filtering::{SobelExtendedFilterImage, centered_filter_geometry},
        },
    },
    types::MaskSize,
};

mod scharr_methods;
mod sobel_border_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: SobelExtendedFilterImage<T, L>,
{
    /// Applies the native horizontal Sobel filter.
    /// `mask_size` selects the native square mask; NPP determines which mask variants are supported.
    /// Allocates a new owned image over the valid source window for the selected square mask.
    /// Each output dimension is the input dimension minus the mask dimension plus one;
    /// the source image must contain the entire mask.
    /// For AC4, copies the corresponding source window first to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Selecting the source window, allocating output, constructing views, or preserving source
    /// alpha can also fail.
    pub fn filter_sobel_horizontal_mask(self, mask_size: MaskSize) -> Result<Self> {
        self.sobel_extended_filter(
            mask_size,
            <Self as SobelExtendedFilterImage<T, L>>::filter_sobel_horizontal_mask_image,
        )
    }

    /// Applies the native vertical Sobel filter.
    /// `mask_size` selects the native square mask; NPP determines which mask variants are supported.
    /// Allocates a new owned image over the valid source window for the selected square mask.
    /// Each output dimension is the input dimension minus the mask dimension plus one;
    /// the source image must contain the entire mask.
    /// For AC4, copies the corresponding source window first to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Selecting the source window, allocating output, constructing views, or preserving source
    /// alpha can also fail.
    pub fn filter_sobel_vertical_mask(self, mask_size: MaskSize) -> Result<Self> {
        self.sobel_extended_filter(
            mask_size,
            <Self as SobelExtendedFilterImage<T, L>>::filter_sobel_vertical_mask_image,
        )
    }

    /// Applies the native horizontal second-derivative Sobel filter.
    /// `mask_size` selects the native square mask; NPP determines which mask variants are supported.
    /// Allocates a new owned image over the valid source window for the selected square mask.
    /// Each output dimension is the input dimension minus the mask dimension plus one;
    /// the source image must contain the entire mask.
    /// For AC4, copies the corresponding source window first to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Selecting the source window, allocating output, constructing views, or preserving source
    /// alpha can also fail.
    pub fn filter_sobel_horizontal_second(self, mask_size: MaskSize) -> Result<Self> {
        self.sobel_extended_filter(
            mask_size,
            <Self as SobelExtendedFilterImage<T, L>>::filter_sobel_horizontal_second_image,
        )
    }

    /// Applies the native vertical second-derivative Sobel filter.
    /// `mask_size` selects the native square mask; NPP determines which mask variants are supported.
    /// Allocates a new owned image over the valid source window for the selected square mask.
    /// Each output dimension is the input dimension minus the mask dimension plus one;
    /// the source image must contain the entire mask.
    /// For AC4, copies the corresponding source window first to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Selecting the source window, allocating output, constructing views, or preserving source
    /// alpha can also fail.
    pub fn filter_sobel_vertical_second(self, mask_size: MaskSize) -> Result<Self> {
        self.sobel_extended_filter(
            mask_size,
            <Self as SobelExtendedFilterImage<T, L>>::filter_sobel_vertical_second_image,
        )
    }

    /// Applies the native cross Sobel filter.
    /// `mask_size` selects the native square mask; NPP determines which mask variants are supported.
    /// Allocates a new owned image over the valid source window for the selected square mask.
    /// Each output dimension is the input dimension minus the mask dimension plus one;
    /// the source image must contain the entire mask.
    /// For AC4, copies the corresponding source window first to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Selecting the source window, allocating output, constructing views, or preserving source
    /// alpha can also fail.
    pub fn filter_sobel_cross(self, mask_size: MaskSize) -> Result<Self> {
        self.sobel_extended_filter(
            mask_size,
            <Self as SobelExtendedFilterImage<T, L>>::filter_sobel_cross_image,
        )
    }

    fn sobel_extended_filter(
        self,
        mask_size: MaskSize,
        filter: SobelExtendedFilterCallback<L, T>,
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

type SobelExtendedFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageWindow<'_, T, L>,
    &mut ImageOutputView<'_, T, L>,
    MaskSize,
) -> Result<()>;
