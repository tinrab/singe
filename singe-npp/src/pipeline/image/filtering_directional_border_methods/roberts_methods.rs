use crate::{
    error::Result,
    image::{SupportedImage, view::ChannelLayout},
    pipeline::image::{ImagePipeline, filtering::EdgeDirectionalBorderFilterImage},
    types::{BorderType, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: EdgeDirectionalBorderFilterImage<T, L>,
{
    /// Applies the native down Roberts filter with border control.
    /// Allocates a new owned image covering the source window from `source_offset` to the
    /// full image's lower-right corner. Native border handling uses `border_type`.
    /// For AC4, copies the corresponding source pixels before filtering to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// Selecting the output window, allocating the image, constructing views, or preserving
    /// source alpha can also fail.
    pub fn filter_roberts_down_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.edge_directional_border_filter(
            source_offset,
            border_type,
            <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_roberts_down_border_image,
        )
    }

    /// Applies the native up Roberts filter with border control.
    /// Allocates a new owned image covering the source window from `source_offset` to the
    /// full image's lower-right corner. Native border handling uses `border_type`.
    /// For AC4, copies the corresponding source pixels before filtering to preserve source alpha.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native filtering, or stream synchronization fail.
    /// Selecting the output window, allocating the image, constructing views, or preserving
    /// source alpha can also fail.
    pub fn filter_roberts_up_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.edge_directional_border_filter(
            source_offset,
            border_type,
            <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_roberts_up_border_image,
        )
    }
}
