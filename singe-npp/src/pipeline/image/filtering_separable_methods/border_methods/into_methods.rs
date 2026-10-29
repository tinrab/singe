use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView},
    pipeline::image::{ImagePipeline, filtering::IntegerSeparableBorderFilterImage},
    types::{BorderType, Point},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: IntegerSeparableBorderFilterImage<T, L>,
{
    /// Applies a one-dimensional vertical filter with `i32` coefficients.
    /// Supply `kernel` in the reversed coefficient order required by NPP. `anchor` gives the
    /// zero-based kernel origin along the filtered axis. The result retains the image sample
    /// type. Passes the slice address directly as a device pointer, without uploading it;
    /// the coefficient storage must also be GPU-readable in the selected CUDA context.
    /// Divides the convolved sum by `divisor` before conversion to the output sample type.
    /// The wrapper forwards `divisor` without validation.
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions select its extent.
    /// The wrapper forwards `border_type` to NPP, which may reject unsupported values.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The kernel must be nonempty, its length must fit `i32`, and `anchor` must be in
    /// `0..kernel.len()`.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    pub fn filter_column_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as IntegerSeparableBorderFilterImage<T, L>>::filter_column_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            kernel,
            anchor,
            divisor,
            border_type,
        )
    }

    /// Applies a one-dimensional horizontal filter with `i32` coefficients.
    /// Supply `kernel` in the reversed coefficient order required by NPP. `anchor` gives the
    /// zero-based kernel origin along the filtered axis. The result retains the image sample
    /// type. Passes the slice address directly as a device pointer, without uploading it;
    /// the coefficient storage must also be GPU-readable in the selected CUDA context.
    /// Divides the convolved sum by `divisor` before conversion to the output sample type.
    /// The wrapper forwards `divisor` without validation.
    /// `source_offset` locates the output ROI in the full source image. The destination
    /// dimensions select its extent.
    /// The wrapper forwards `border_type` to NPP, which may reject unsupported values.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// The kernel must be nonempty, its length must fit `i32`, and `anchor` must be in
    /// `0..kernel.len()`.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    pub fn filter_row_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as IntegerSeparableBorderFilterImage<T, L>>::filter_row_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            kernel,
            anchor,
            divisor,
            border_type,
        )
    }
}
