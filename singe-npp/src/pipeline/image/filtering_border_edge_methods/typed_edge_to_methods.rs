use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageWindow},
    pipeline::image::{ImagePipeline, filtering::TypedEdgeDirectionalFilterImage},
    types::MaskSize,
};

mod border_to_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Applies the native horizontal Sobel filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses the selected `mask_size`; NPP may reject unsupported masks.
    /// The source window must retain a complete parent-image halo for the centered
    /// neighborhood, and source-window and destination pixel dimensions must match.
    /// The wrapper requires a square mask and uses its center as the anchor.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source window must match the destination dimensions and retain the required halo.
    /// One-dimensional mask variants are rejected.
    pub fn filter_sobel_horizontal_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_sobel_horizontal_to_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }

    /// Applies the native vertical Sobel filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses the selected `mask_size`; NPP may reject unsupported masks.
    /// The source window must retain a complete parent-image halo for the centered
    /// neighborhood, and source-window and destination pixel dimensions must match.
    /// The wrapper requires a square mask and uses its center as the anchor.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source window must match the destination dimensions and retain the required halo.
    /// One-dimensional mask variants are rejected.
    pub fn filter_sobel_vertical_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, M>,
        mask_size: MaskSize,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_sobel_vertical_to_image(
            stream_context,
            source,
            destination,
            mask_size,
        )
    }

    /// Applies the native horizontal Scharr filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses a centered 3-by-3 neighborhood.
    /// The source window must retain a complete parent-image halo for the centered
    /// neighborhood, and source-window and destination pixel dimensions must match.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source window must match the destination dimensions and retain the required halo.
    pub fn filter_scharr_horizontal_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, M>,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_scharr_horizontal_to_image(
            stream_context,
            source,
            destination,
        )
    }

    /// Applies the native vertical Scharr filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses a centered 3-by-3 neighborhood.
    /// The source window must retain a complete parent-image halo for the centered
    /// neighborhood, and source-window and destination pixel dimensions must match.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source window must match the destination dimensions and retain the required halo.
    pub fn filter_scharr_vertical_to_into<D, M>(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, D, M>,
    ) -> Result<()>
    where
        D: Copy,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_scharr_vertical_to_image(
            stream_context,
            source,
            destination,
        )
    }
}
