use crate::{
    error::Result,
    image::{SupportedImage, view::ChannelLayout},
    pipeline::image::{ImagePipeline, filtering::SobelExtendedBorderFilterImage},
    types::{BorderType, MaskSize, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: SobelExtendedBorderFilterImage<T, L>,
{
    /// Applies the native horizontal second-derivative Sobel filter with border control.
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
    pub fn filter_sobel_horizontal_second_border(
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
            >>::filter_sobel_horizontal_second_border_image,
        )
    }

    /// Applies the native vertical second-derivative Sobel filter with border control.
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
    pub fn filter_sobel_vertical_second_border(
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
            >>::filter_sobel_vertical_second_border_image,
        )
    }

    /// Applies the native cross Sobel filter with border control.
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
    pub fn filter_sobel_cross_border(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.sobel_extended_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as SobelExtendedBorderFilterImage<T, L>>::filter_sobel_cross_border_image,
        )
    }
}
