use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageWindow},
    },
    pipeline::{
        ImageBacking,
        image::{
            ImagePipeline,
            filtering::{BoxFilterImage, DoubleKernelFilterImage},
        },
    },
    types::{Point, Size},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: BoxFilterImage<T, L>,
{
    /// Applies a rectangular averaging filter to the source ROI.
    ///
    /// The source window and destination must have matching pixel
    /// dimensions. The source window must retain the complete parent-image halo required
    /// by the mask and anchor.
    /// Mask dimensions must be positive and `anchor` must lie inside the mask.
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
    /// Mask dimensions must be positive and the anchor must lie within the mask.
    pub fn filter_box_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: Size,
        anchor: Point,
    ) -> Result<()> {
        <Self as BoxFilterImage<T, L>>::filter_box_image(
            stream_context,
            source,
            destination,
            mask_size,
            anchor,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: DoubleKernelFilterImage<T, L>,
{
    /// Applies an `f64` convolution kernel to the source ROI.
    ///
    /// The provided implementation filters C1 `f64` images. The source window and destination must have matching pixel
    /// dimensions. The source window must retain the complete parent-image halo required
    /// by the mask and anchor.
    /// Supply coefficients in the reverse order required by NPP. Kernel dimensions must
    /// be positive, their checked product must equal `kernel.len()`, and `anchor` must lie
    /// inside the kernel. The coefficient slice address is passed directly to NPP without
    /// upload, context validation, or producer waits. Coefficients must satisfy Rust slice
    /// requirements, be GPU-readable in the selected context, and be ready for native reads
    /// through stream completion.
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
    pub fn filter_kernel64f_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f64],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<()> {
        <Self as DoubleKernelFilterImage<T, L>>::filter_kernel64f_image(
            stream_context,
            source,
            destination,
            kernel,
            kernel_size,
            anchor,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: DoubleKernelFilterImage<T, L>,
{
    /// Applies an `f64` convolution kernel to the source ROI.
    ///
    /// The provided implementation filters C1 `f64` images. Uses the valid source window, reducing width and height by
    /// the kernel width and height minus one. `anchor` selects its position within the
    /// parent source image.
    /// Supply coefficients in the reverse order required by NPP. Kernel dimensions must
    /// be positive, their checked product must equal `kernel.len()`, and `anchor` must lie
    /// inside the kernel. The coefficient slice address is passed directly to NPP without
    /// upload, context validation, or producer waits. Coefficients must satisfy Rust slice
    /// requirements, be GPU-readable in the selected context, and be ready for native reads
    /// through stream completion.
    /// Allocates a new owned image for the valid source window.
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
    /// Source or destination view construction, valid-window geometry, or output allocation can also fail.
    pub fn filter_kernel64f(
        self,
        kernel: &[f64],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?.valid_window(kernel_size, anchor)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&source.view())?;
            }
            <Self as DoubleKernelFilterImage<T, L>>::filter_kernel64f_image(
                self.stream_context,
                &source,
                &mut destination_view,
                kernel,
                kernel_size,
                anchor,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }
}
