use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::filtering_separable_methods::{
        ImageBacking,
        border_methods::{FloatSeparableBorderFilterImage, ImagePipeline},
    },
    types::{BorderType, Point},
};

mod into_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: FloatSeparableBorderFilterImage<T, L>,
{
    /// Applies a one-dimensional vertical filter with `f32` coefficients.
    /// Supply `kernel` in the reversed coefficient order required by NPP. `anchor` gives the
    /// zero-based kernel origin along the filtered axis. The result retains the image sample
    /// type. Passes the slice address directly as a device pointer, without uploading it;
    /// the coefficient storage must also be GPU-readable in the selected CUDA context.
    /// `source_offset` locates the output ROI in the full source image. Allocates a new
    /// owned image spanning from that offset to the lower-right source edge.
    /// The wrapper forwards `border_type` to NPP, which may reject unsupported values.
    /// Preserves AC4 alpha from the corresponding source window.
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
    /// Image allocation or view construction, including alpha preservation copies, can also fail.
    pub fn filter_column_border32f(
        self,
        source_offset: Point,
        kernel: &[f32],
        anchor: i32,
        border_type: BorderType,
    ) -> Result<Self> {
        self.float_separable_border_filter(
            source_offset,
            kernel,
            anchor,
            border_type,
            <Self as FloatSeparableBorderFilterImage<T, L>>::filter_column_border32f_image,
        )
    }

    /// Applies a one-dimensional horizontal filter with `f32` coefficients.
    /// Supply `kernel` in the reversed coefficient order required by NPP. `anchor` gives the
    /// zero-based kernel origin along the filtered axis. The result retains the image sample
    /// type. Passes the slice address directly as a device pointer, without uploading it;
    /// the coefficient storage must also be GPU-readable in the selected CUDA context.
    /// `source_offset` locates the output ROI in the full source image. Allocates a new
    /// owned image spanning from that offset to the lower-right source edge.
    /// The wrapper forwards `border_type` to NPP, which may reject unsupported values.
    /// Preserves AC4 alpha from the corresponding source window.
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
    /// Image allocation or view construction, including alpha preservation copies, can also fail.
    pub fn filter_row_border32f(
        self,
        source_offset: Point,
        kernel: &[f32],
        anchor: i32,
        border_type: BorderType,
    ) -> Result<Self> {
        self.float_separable_border_filter(
            source_offset,
            kernel,
            anchor,
            border_type,
            <Self as FloatSeparableBorderFilterImage<T, L>>::filter_row_border32f_image,
        )
    }

    fn float_separable_border_filter(
        self,
        source_offset: Point,
        kernel: &[f32],
        anchor: i32,
        border_type: BorderType,
        filter: FloatSeparableBorderFilterCallback<L, T>,
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
            filter(
                self.stream_context,
                &source,
                source_offset,
                &mut destination_view,
                kernel,
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
}

type FloatSeparableBorderFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    Point,
    &mut ImageOutputView<'_, T, L>,
    &[f32],
    i32,
    BorderType,
) -> Result<()>;
