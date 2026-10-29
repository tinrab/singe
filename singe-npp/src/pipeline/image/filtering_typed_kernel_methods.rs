use std::sync::Arc;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image,
        view::{ChannelLayout, ImageOutputView, ImageWindow},
    },
    pipeline::image::{ImageBacking, ImagePipeline, filtering::TypedFloatKernelFilterImage},
    types::{Point, Size},
};

mod border_methods;

impl<S, SL> ImagePipeline<'_, S, SL>
where
    S: Copy,
    SL: ChannelLayout,
{
    /// Applies an `f32` kernel with native sample conversion to the source ROI.
    ///
    /// The provided implementations convert `u8` or `i8` to `i16` while retaining the
    /// C1, C3, C4, or AC4 layout. The source window and destination must have matching pixel
    /// dimensions. The source window must retain the complete parent-image halo required
    /// by the mask and anchor.
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
    /// The source window must retain the required halo and match destination dimensions.
    /// Kernel dimensions must be positive, their checked area must equal the coefficient
    /// count, and the anchor must lie within the kernel.
    pub fn filter_kernel32f_to_into<D, DL>(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, S, SL>,
        destination: &mut ImageOutputView<'_, D, DL>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<()>
    where
        D: Copy,
        DL: ChannelLayout,
        Self: TypedFloatKernelFilterImage<S, SL, D, DL>,
    {
        <Self as TypedFloatKernelFilterImage<S, SL, D, DL>>::filter_kernel32f_to_image(
            stream_context,
            source,
            destination,
            kernel,
            kernel_size,
            anchor,
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
    /// C1, C3, C4, or AC4 layout. Uses the valid source window, reducing width and height by
    /// the kernel width and height minus one. `anchor` selects its position within the
    /// parent source image.
    /// Supply coefficients in the reverse order required by NPP. Kernel dimensions must
    /// be positive, their checked product must equal `kernel.len()`, and `anchor` must lie
    /// inside the kernel. The coefficient slice address is passed directly to NPP without
    /// upload, context validation, or producer waits. Coefficients must satisfy Rust slice
    /// requirements, be GPU-readable in the selected context, and be ready for native reads
    /// through stream completion.
    /// Takes ownership of the supplied initialized `destination` image and returns it in
    /// the resulting pipeline; no new output image is allocated. Its dimensions must match
    /// the valid source window. Any ignored destination alpha is preserved.
    /// Prepares source and destination regions, waits for their producers on other streams,
    /// rejects capture, and returns after successful stream completion. Row padding remains
    /// unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native filtering, or stream synchronization fail.
    /// The source window must retain the required halo and match destination dimensions.
    /// Kernel dimensions must be positive, their checked area must equal the coefficient
    /// count, and the anchor must lie within the kernel.
    /// Source or destination view construction, valid-window geometry, or capture-time
    /// allocation retention can also fail.
    pub fn filter_kernel32f_to<D, DL>(
        self,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
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
        let source = self.view()?.valid_window(kernel_size, anchor)?;
        {
            let mut output = destination.output_view()?;
            <Self as TypedFloatKernelFilterImage<S, SL, D, DL>>::filter_kernel32f_to_image(
                self.stream_context,
                &source,
                &mut output,
                kernel,
                kernel_size,
                anchor,
            )?;
        }
        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
