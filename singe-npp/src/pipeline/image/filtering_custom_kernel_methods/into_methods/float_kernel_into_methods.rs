use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView, ImageWindow},
    pipeline::image::filtering_custom_kernel_methods::into_methods::{
        FloatKernelFilterImage, ImagePipeline,
    },
    types::{BorderType, Point, Size},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: FloatKernelFilterImage<T, L>,
{
    /// Filters each active channel using a rectangular floating-point coefficient kernel.
    /// Supply coefficients in the reverse order required by NPP. `kernel_size` gives the
    /// width and height, and `anchor` locates the kernel origin relative to the source pixel.
    /// Passes the slice address directly as a native device pointer, without uploading it;
    /// its storage must also be GPU-readable in the selected CUDA context.
    /// The source window must retain a complete parent-image halo for `kernel_size` and
    /// `anchor`. Source-window and destination dimensions must match.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`,
    /// and `anchor` must lie within the kernel.
    /// The source window must match the destination size and retain the required halo.
    pub fn filter_kernel32f_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<()> {
        <Self as FloatKernelFilterImage<T, L>>::filter_kernel32f_image(
            stream_context,
            source,
            destination,
            kernel,
            kernel_size,
            anchor,
        )
    }

    /// Filters each active channel using a rectangular floating-point coefficient kernel.
    /// Supply coefficients in the reverse order required by NPP. `kernel_size` gives the
    /// width and height, and `anchor` locates the kernel origin relative to the source pixel.
    /// Passes the slice address directly as a native device pointer, without uploading it;
    /// its storage must also be GPU-readable in the selected CUDA context.
    /// `source_offset` locates the output ROI in the full source. The destination dimensions
    /// select its extent.
    /// AC4 destination alpha retains its previous initialization state; this call does not
    /// initialize alpha.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`,
    /// and `anchor` must lie within the kernel.
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    pub fn filter_kernel32f_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as FloatKernelFilterImage<T, L>>::filter_kernel32f_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            kernel,
            kernel_size,
            anchor,
            border_type,
        )
    }
}
