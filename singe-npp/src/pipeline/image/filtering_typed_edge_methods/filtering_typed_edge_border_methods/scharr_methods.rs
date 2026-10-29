use crate::{
    error::Result,
    image::{SupportedImage, view::ChannelLayout},
    pipeline::image::{ImagePipeline, filtering::TypedEdgeDirectionalBorderFilterImage},
    types::{BorderType, Point},
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
    /// `source_offset` locates the output ROI in the full source image. Allocates a new
    /// owned image spanning from that offset to the lower-right source edge.
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
    /// Output allocation or image view construction can also fail.
    pub fn filter_scharr_horizontal_border_to<D, M>(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy + SupportedImage<M>,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalBorderFilterImage<T, L, D, M>,
    {
        self.typed_edge_directional_border_filter(
            source_offset,
            border_type,
            <Self as TypedEdgeDirectionalBorderFilterImage<
                T,
                L,
                D,
                M,
            >>::filter_scharr_horizontal_border_to_image,
        )
    }

    /// Applies the native vertical Scharr filter into the selected output type and layout.
    /// The provided typed implementations convert C1 `u8` or `i8` source samples to C1
    /// `i16` output samples.
    /// Uses a centered 3-by-3 neighborhood.
    /// `source_offset` locates the output ROI in the full source image. Allocates a new
    /// owned image spanning from that offset to the lower-right source edge.
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
    /// Output allocation or image view construction can also fail.
    pub fn filter_scharr_vertical_border_to<D, M>(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy + SupportedImage<M>,
        M: ChannelLayout,
        Self: TypedEdgeDirectionalBorderFilterImage<T, L, D, M>,
    {
        self.typed_edge_directional_border_filter(
            source_offset,
            border_type,
            <Self as TypedEdgeDirectionalBorderFilterImage<
                T,
                L,
                D,
                M,
            >>::filter_scharr_vertical_border_to_image,
        )
    }
}
