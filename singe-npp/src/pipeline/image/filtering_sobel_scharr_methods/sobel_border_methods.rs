use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{ImageBacking, ImagePipeline, filtering::SobelExtendedBorderFilterImage},
    types::{BorderType, MaskSize, Point},
};

mod second_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: SobelExtendedBorderFilterImage<T, L>,
{
    /// Applies the native horizontal Sobel filter with border control.
    /// `mask_size` selects the native square mask; NPP determines which mask variants are supported.
    /// Allocates a new owned image covering the source window from `source_offset` to the
    /// image's lower-right corner. Border handling uses `border_type`.
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
    pub fn filter_sobel_horizontal_mask_border(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.sobel_extended_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as SobelExtendedBorderFilterImage<
                T,
                L,
            >>::filter_sobel_horizontal_mask_border_image,
        )
    }

    /// Applies the native vertical Sobel filter with border control.
    /// `mask_size` selects the native square mask; NPP determines which mask variants are supported.
    /// Allocates a new owned image covering the source window from `source_offset` to the
    /// image's lower-right corner. Border handling uses `border_type`.
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
    pub fn filter_sobel_vertical_mask_border(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.sobel_extended_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as SobelExtendedBorderFilterImage<T, L>>::filter_sobel_vertical_mask_border_image,
        )
    }

    pub(crate) fn sobel_extended_border_filter(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
        filter: SobelExtendedBorderFilterCallback<L, T>,
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

type SobelExtendedBorderFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    Point,
    &mut ImageOutputView<'_, T, L>,
    MaskSize,
    BorderType,
) -> Result<()>;
