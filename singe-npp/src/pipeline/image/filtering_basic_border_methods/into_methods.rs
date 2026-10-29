use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView},
    pipeline::image::{ImagePipeline, filtering::HighLowGaussBorderFilterImage},
    types::{BorderType, MaskSize, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: HighLowGaussBorderFilterImage<T, L>,
{
    /// Applies the native high-pass filter with source border control.
    ///
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions select its extent.
    /// Forwards `mask_size` and `border_type` to NPP, which may reject unsupported values.
    /// For AC4, writes the three color channels and leaves destination alpha untouched.
    /// Row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source offset must address a source pixel; destination-sized ROI endpoints must
    /// not overflow or exceed the source image.
    pub fn filter_high_pass_border_partial_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as HighLowGaussBorderFilterImage<T, L>>::filter_high_pass_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }

    /// Applies the native low-pass filter with source border control.
    ///
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions select its extent.
    /// Forwards `mask_size` and `border_type` to NPP, which may reject unsupported values.
    /// For AC4, writes the three color channels and leaves destination alpha untouched.
    /// Row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source offset must address a source pixel; destination-sized ROI endpoints must
    /// not overflow or exceed the source image.
    pub fn filter_low_pass_border_partial_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as HighLowGaussBorderFilterImage<T, L>>::filter_low_pass_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }

    /// Applies the native Gaussian filter with source border control.
    ///
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions select its extent.
    /// Forwards `mask_size` and `border_type` to NPP, which may reject unsupported values.
    /// For AC4, writes the three color channels and leaves destination alpha untouched.
    /// Row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source offset must address a source pixel; destination-sized ROI endpoints must
    /// not overflow or exceed the source image.
    pub fn filter_gauss_border_partial_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as HighLowGaussBorderFilterImage<T, L>>::filter_gauss_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            mask_size,
            border_type,
        )
    }
}
