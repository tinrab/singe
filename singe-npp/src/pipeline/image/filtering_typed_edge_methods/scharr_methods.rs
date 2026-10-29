use crate::{
    error::Result,
    image::{SupportedImage, view::ChannelLayout},
    pipeline::image::{ImagePipeline, filtering::TypedEdgeDirectionalFilterImage},
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Applies the native horizontal Scharr filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses a centered 3-by-3 neighborhood.
    /// Allocates a new owned output for the valid source window. Reduces width and height
    /// by two pixels.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source must contain the complete neighborhood and its valid output window.
    /// Output allocation or image view construction can also fail.
    pub fn filter_scharr_horizontal_to<D, M>(self) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy + SupportedImage<M>,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        self.typed_edge_directional_filter(
            <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_scharr_horizontal_to_image,
        )
    }

    /// Applies the native vertical Scharr filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses a centered 3-by-3 neighborhood.
    /// Allocates a new owned output for the valid source window. Reduces width and height
    /// by two pixels.
    /// Prepares source and destination regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Writes every logical output
    /// sample; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filter execution, or stream synchronization fail.
    /// The source must contain the complete neighborhood and its valid output window.
    /// Output allocation or image view construction can also fail.
    pub fn filter_scharr_vertical_to<D, M>(self) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy + SupportedImage<M>,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        self.typed_edge_directional_filter(
            <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_scharr_vertical_to_image,
        )
    }
}
