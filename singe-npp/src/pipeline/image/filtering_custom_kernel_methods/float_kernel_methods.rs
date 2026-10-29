use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageWindow},
    },
    pipeline::image::{
        ImageBacking,
        filtering_custom_kernel_methods::{FloatKernelFilterImage, ImagePipeline},
    },
    types::{BorderType, Point, Size},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: FloatKernelFilterImage<T, L>,
{
    /// Filters each active channel using a rectangular floating-point coefficient kernel.
    /// Supply coefficients in the reverse order required by NPP. `kernel_size` gives the
    /// width and height, and `anchor` locates the kernel origin relative to the source pixel.
    /// Passes the slice address directly as a native device pointer, without uploading it;
    /// its storage must also be GPU-readable in the selected CUDA context.
    /// Allocates a new owned image for the valid source window, reducing width and height
    /// by `kernel_size.width - 1` and `kernel_size.height - 1` respectively.
    /// Preserves AC4 alpha from the corresponding source window.
    /// Prepares participating image regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or image validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Kernel dimensions must be positive, their checked product must equal `kernel.len()`,
    /// and `anchor` must lie within the kernel.
    /// The source must contain the complete kernel and its valid output window.
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_kernel32f(
        self,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<Self> {
        self.float_kernel_filter(
            kernel,
            kernel_size,
            anchor,
            <Self as FloatKernelFilterImage<T, L>>::filter_kernel32f_image,
        )
    }

    /// Filters each active channel using a rectangular floating-point coefficient kernel.
    /// Supply coefficients in the reverse order required by NPP. `kernel_size` gives the
    /// width and height, and `anchor` locates the kernel origin relative to the source pixel.
    /// Passes the slice address directly as a native device pointer, without uploading it;
    /// its storage must also be GPU-readable in the selected CUDA context.
    /// `source_offset` locates the output ROI in the full source. Allocates a new owned
    /// output spanning from that offset to the lower-right source edge.
    /// Preserves AC4 alpha from the corresponding source window.
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
    /// Image allocation, view construction, or alpha preservation copies can also fail.
    pub fn filter_kernel32f_border(
        self,
        source_offset: Point,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let source = self.view()?;
        let output_source = source.window_from(source_offset)?;
        let mut destination =
            Image::<T, L>::create_uninit(self.stream_context.cuda_context(), output_source.size())?;

        {
            let mut destination_view = destination.output()?;
            if L::IGNORES_ALPHA {
                destination_view.copy_from(&output_source.view())?;
            }
            <Self as FloatKernelFilterImage<T, L>>::filter_kernel32f_border_image(
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

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            // SAFETY: the completed operation above wrote every element.
            backing: ImageBacking::Owned(unsafe { destination.assume_init() }),
        })
    }

    fn float_kernel_filter(
        self,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        filter: FloatKernelFilterCallback<L, T>,
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
            filter(
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

type FloatKernelFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageWindow<'_, T, L>,
    &mut ImageOutputView<'_, T, L>,
    &[f32],
    Size,
    Point,
) -> Result<()>;
