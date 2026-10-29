use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageOutputView, ImageView, ImageWindow},
    pipeline::image::{
        ImagePipeline,
        filtering::{
            DirectionalBorderFilterImage, FloatKernelFilterImage, IntegerKernelFilterImage,
            UnsharpBorderFilterImage,
        },
    },
    types::{BorderType, Point, Size},
};

mod float_kernel_into_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: IntegerKernelFilterImage<T, L>,
{
    /// Filters each active channel using a rectangular integer coefficient kernel.
    /// Supply coefficients in the reverse order required by NPP. `kernel_size` gives the
    /// width and height, and `anchor` locates the kernel origin relative to the source pixel.
    /// Passes the slice address directly as a native device pointer, without uploading it;
    /// its storage must also be GPU-readable in the selected CUDA context.
    /// Divides the convolved sum by `divisor` before converting to the destination sample
    /// type. The wrapper forwards `divisor` without validation.
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
    pub fn filter_kernel_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[i32],
        kernel_size: Size,
        anchor: Point,
        divisor: i32,
    ) -> Result<()> {
        <Self as IntegerKernelFilterImage<T, L>>::filter_kernel_image(
            stream_context,
            source,
            destination,
            kernel,
            kernel_size,
            anchor,
            divisor,
        )
    }

    /// Filters each active channel using a rectangular integer coefficient kernel.
    /// Supply coefficients in the reverse order required by NPP. `kernel_size` gives the
    /// width and height, and `anchor` locates the kernel origin relative to the source pixel.
    /// Passes the slice address directly as a native device pointer, without uploading it;
    /// its storage must also be GPU-readable in the selected CUDA context.
    /// Divides the convolved sum by `divisor` before converting to the destination sample
    /// type. The wrapper forwards `divisor` without validation.
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
    pub fn filter_kernel_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[i32],
        kernel_size: Size,
        anchor: Point,
        divisor: i32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as IntegerKernelFilterImage<T, L>>::filter_kernel_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            kernel,
            kernel_size,
            anchor,
            divisor,
            border_type,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: DirectionalBorderFilterImage<T, L>,
{
    /// Applies the native sharpening filter with border control.
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
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    pub fn filter_sharpen_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as DirectionalBorderFilterImage<T, L>>::filter_sharpen_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            border_type,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: UnsharpBorderFilterImage<T, L>,
{
    /// Sharpens the image with a Gaussian unsharp mask and border control.
    /// `radius` is the Gaussian radius in pixels, excluding the center; `sigma` is its
    /// standard deviation in pixels. Adds `weight * (source - blurred)` where the absolute
    /// difference is at least `threshold`. `weight` and `threshold` are forwarded unchecked.
    /// The native routine supports replicated borders; the wrapper forwards `border_type`
    /// without validating it. Queries and allocates temporary device scratch.
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
    /// The source offset must select a source pixel, and the destination-sized ROI must fit
    /// inside the source without coordinate overflow.
    /// `radius` and `sigma` must compare greater than zero. The scratch-size query and device
    /// scratch allocation can also fail.
    pub fn filter_unsharp_border_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        radius: f32,
        sigma: f32,
        weight: f32,
        threshold: f32,
        border_type: BorderType,
    ) -> Result<()> {
        <Self as UnsharpBorderFilterImage<T, L>>::filter_unsharp_border_image(
            stream_context,
            source,
            source_offset,
            destination,
            radius,
            sigma,
            weight,
            threshold,
            border_type,
        )
    }
}
