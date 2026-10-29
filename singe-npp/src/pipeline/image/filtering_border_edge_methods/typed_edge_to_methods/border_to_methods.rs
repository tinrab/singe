use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView},
    pipeline::image::{ImagePipeline, filtering::TypedEdgeDirectionalBorderFilterImage},
    types::{BorderType, MaskSize, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Applies the native horizontal Sobel filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses the selected `mask_size`; NPP may reject unsupported masks.
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions select its extent.
    /// The wrapper forwards `border_type` to NPP, which may reject unsupported values.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source offset must select a source pixel and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    pub fn filter_sobel_horizontal_border_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalBorderFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalBorderFilterImage<
            T,
            L,
            D,
            M,
        >>::filter_sobel_horizontal_border_to_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }

    /// Applies the native vertical Sobel filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses the selected `mask_size`; NPP may reject unsupported masks.
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions select its extent.
    /// The wrapper forwards `border_type` to NPP, which may reject unsupported values.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source offset must select a source pixel and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    pub fn filter_sobel_vertical_border_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalBorderFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalBorderFilterImage<
            T,
            L,
            D,
            M,
        >>::filter_sobel_vertical_border_to_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }

    /// Applies the native horizontal Scharr filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses a centered 3-by-3 neighborhood.
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions select its extent.
    /// The wrapper forwards `border_type` to NPP, which may reject unsupported values.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source offset must select a source pixel and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    pub fn filter_scharr_horizontal_border_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, D, M>,
        border_type: BorderType,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalBorderFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalBorderFilterImage<
            T,
            L,
            D,
            M,
        >>::filter_scharr_horizontal_border_to_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }

    /// Applies the native vertical Scharr filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses a centered 3-by-3 neighborhood.
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions select its extent.
    /// The wrapper forwards `border_type` to NPP, which may reject unsupported values.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source offset must select a source pixel and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    pub fn filter_scharr_vertical_border_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, D, M>,
        border_type: BorderType,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalBorderFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalBorderFilterImage<
            T,
            L,
            D,
            M,
        >>::filter_scharr_vertical_border_to_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}
