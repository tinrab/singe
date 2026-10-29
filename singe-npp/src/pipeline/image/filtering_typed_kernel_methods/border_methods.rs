use std::sync::Arc;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::filtering_typed_kernel_methods::{
        ImageBacking, ImagePipeline, TypedFloatKernelFilterImage,
    },
    types::{BorderType, Point, Size},
};

impl<S, SL> ImagePipeline<'_, S, SL>
where
    S: Copy,
    SL: ChannelLayout,
{
    /// Applies an `f32` kernel with native sample conversion to the source ROI.
    ///
    /// The provided implementations convert `u8` or `i8` to `i16` while retaining the
    /// C1, C3, C4, or AC4 layout. `source_offset` locates the destination-sized ROI within the
    /// full source image. `border_type` is forwarded to NPP, which may reject unsupported
    /// values.
    /// Supply coefficients in the reverse order required by NPP. Kernel dimensions must
    /// be positive, their checked product must equal `kernel.len()`, and `anchor` must lie
    /// inside the kernel. The coefficient slice address is passed directly to NPP without
    /// upload, context validation, or producer waits. Coefficients must satisfy Rust slice
    /// requirements, be GPU-readable in the selected context, and be ready for native reads
    /// through stream completion.
    /// For AC4, writes the color channels and leaves destination alpha untouched.
    /// Prepares source and destination regions, waits for their producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native filtering, or stream synchronization fail.
    /// The source offset must address a source pixel; destination-sized ROI endpoints must
    /// not overflow or exceed the source image.
    /// Kernel dimensions must be positive, their checked area must equal the coefficient
    /// count, and the anchor must lie within the kernel.
    pub fn filter_kernel32f_border_to_into<D, DL>(
        stream_context: &StreamContext,
        source: &ImageView<'_, S, SL>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, D, DL>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>
    where
        D: Copy,
        DL: ChannelLayout,
        Self: TypedFloatKernelFilterImage<S, SL, D, DL>,
    {
        <Self as TypedFloatKernelFilterImage<S, SL, D, DL>>::filter_kernel32f_border_to_image(
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

impl<'a, S, SL> ImagePipeline<'a, S, SL>
where
    S: Copy,
    SL: ChannelLayout,
{
    /// Applies an `f32` kernel with native sample conversion to the source ROI.
    ///
    /// The provided implementations convert `u8` or `i8` to `i16` while retaining the
    /// C1, C3, C4, or AC4 layout. `source_offset` locates the destination-sized ROI within the
    /// full source image. `border_type` is forwarded to NPP, which may reject unsupported
    /// values.
    /// Supply coefficients in the reverse order required by NPP. Kernel dimensions must
    /// be positive, their checked product must equal `kernel.len()`, and `anchor` must lie
    /// inside the kernel. The coefficient slice address is passed directly to NPP without
    /// upload, context validation, or producer waits. Coefficients must satisfy Rust slice
    /// requirements, be GPU-readable in the selected context, and be ready for native reads
    /// through stream completion.
    /// Takes ownership of the supplied initialized `destination` image and returns it in
    /// the resulting pipeline; no new output image is allocated. Its dimensions must match
    /// the requested output ROI. Any ignored destination alpha is preserved.
    /// Prepares source and destination regions, waits for their producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native filtering, or stream synchronization fail.
    /// The source offset must address a source pixel; destination-sized ROI endpoints must
    /// not overflow or exceed the source image.
    /// Kernel dimensions must be positive, their checked area must equal the coefficient
    /// count, and the anchor must lie within the kernel.
    /// Source or destination view construction or capture-time allocation retention can
    /// also fail.
    pub fn filter_kernel32f_border_to<D, DL>(
        self,
        source_offset: Point,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        border_type: BorderType,
        mut destination: Image<D, DL>,
    ) -> Result<ImagePipeline<'a, D, DL>>
    where
        D: Copy,
        DL: ChannelLayout,
        Self: TypedFloatKernelFilterImage<S, SL, D, DL>,
    {
        self.stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(destination.allocation()))?;
        self.ensure_eager()?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output_view()?;
            <Self as TypedFloatKernelFilterImage<S, SL, D, DL>>::filter_kernel32f_border_to_image(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                kernel,
                kernel_size,
                anchor,
                border_type,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
