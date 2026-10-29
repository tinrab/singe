use crate::{
    error::Result,
    image::{SupportedImage, view::ChannelLayout},
    pipeline::image::{ImagePipeline, filtering::TypedSobelExtendedFilterImage},
    types::MaskSize,
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Applies the native horizontal second Sobel filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses the selected `mask_size`; NPP may reject unsupported masks.
    /// Allocates a new owned output for the valid source window. Reduces width and height
    /// by the selected square mask extent minus one.
    /// The wrapper requires a square mask and uses its center as the anchor.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source must contain the complete neighborhood and its valid output window.
    /// One-dimensional mask variants are rejected.
    /// Output allocation or image view construction can also fail.
    pub fn filter_sobel_horizontal_second_to<D, M>(
        self,
        mask_size: MaskSize,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy + SupportedImage<M>,
        M: ChannelLayout,
        Self: TypedSobelExtendedFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_filter(
            mask_size,
            <Self as TypedSobelExtendedFilterImage<
                T,
                L,
                D,
                M,
            >>::filter_sobel_horizontal_second_to_image,
        )
    }

    /// Applies the native vertical second Sobel filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses the selected `mask_size`; NPP may reject unsupported masks.
    /// Allocates a new owned output for the valid source window. Reduces width and height
    /// by the selected square mask extent minus one.
    /// The wrapper requires a square mask and uses its center as the anchor.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source must contain the complete neighborhood and its valid output window.
    /// One-dimensional mask variants are rejected.
    /// Output allocation or image view construction can also fail.
    pub fn filter_sobel_vertical_second_to<D, M>(
        self,
        mask_size: MaskSize,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy + SupportedImage<M>,
        M: ChannelLayout,
        Self: TypedSobelExtendedFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_filter(
            mask_size,
            <Self as TypedSobelExtendedFilterImage<
                T,
                L,
                D,
                M,
            >>::filter_sobel_vertical_second_to_image,
        )
    }

    /// Applies the native cross Sobel filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses the selected `mask_size`; NPP may reject unsupported masks.
    /// Allocates a new owned output for the valid source window. Reduces width and height
    /// by the selected square mask extent minus one.
    /// The wrapper requires a square mask and uses its center as the anchor.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source must contain the complete neighborhood and its valid output window.
    /// One-dimensional mask variants are rejected.
    /// Output allocation or image view construction can also fail.
    pub fn filter_sobel_cross_to<D, M>(self, mask_size: MaskSize) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy + SupportedImage<M>,
        M: ChannelLayout,
        Self: TypedSobelExtendedFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_filter(
            mask_size,
            <Self as TypedSobelExtendedFilterImage<T, L, D, M>>::filter_sobel_cross_to_image,
        )
    }

    /// Applies the native Laplace filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses the selected `mask_size`; NPP may reject unsupported masks.
    /// Allocates a new owned output for the valid source window. Reduces width and height
    /// by the selected square mask extent minus one.
    /// The wrapper requires a square mask and uses its center as the anchor.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source must contain the complete neighborhood and its valid output window.
    /// One-dimensional mask variants are rejected.
    /// Output allocation or image view construction can also fail.
    pub fn filter_laplace_to<D, M>(self, mask_size: MaskSize) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy + SupportedImage<M>,
        M: ChannelLayout,
        Self: TypedSobelExtendedFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_filter(
            mask_size,
            <Self as TypedSobelExtendedFilterImage<T, L, D, M>>::filter_laplace_to_image,
        )
    }
}
