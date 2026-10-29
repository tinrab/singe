use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageWindow},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        filtering::{FilterAxis, IntegerSeparableFilterImage},
    },
};

mod border_methods;

mod sum_window_methods;

mod sum_window_border_methods;

mod double_methods;

mod float_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: IntegerSeparableFilterImage<T, L>,
{
    /// Applies a one-dimensional vertical filter with `i32` coefficients.
    /// Supply `kernel` in the reversed coefficient order required by NPP. `anchor` gives the
    /// zero-based kernel origin along the filtered axis. The result retains the image sample
    /// type. Passes the slice address directly as a device pointer, without uploading it;
    /// the coefficient storage must also be GPU-readable in the selected CUDA context.
    /// Divides the convolved sum by `divisor` before conversion to the output sample type.
    /// The wrapper forwards `divisor` without validation.
    /// The source window must retain the parent-image halo needed for the vertical window
    /// and anchor. Source-window and destination dimensions must match.
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
    /// The source window must match the destination size and retain the required halo.
    pub fn filter_column_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
    ) -> Result<()> {
        <Self as IntegerSeparableFilterImage<T, L>>::filter_column_image(
            stream_context,
            source,
            destination,
            kernel,
            anchor,
            divisor,
        )
    }

    /// Applies a one-dimensional horizontal filter with `i32` coefficients.
    /// Supply `kernel` in the reversed coefficient order required by NPP. `anchor` gives the
    /// zero-based kernel origin along the filtered axis. The result retains the image sample
    /// type. Passes the slice address directly as a device pointer, without uploading it;
    /// the coefficient storage must also be GPU-readable in the selected CUDA context.
    /// Divides the convolved sum by `divisor` before conversion to the output sample type.
    /// The wrapper forwards `divisor` without validation.
    /// The source window must retain the parent-image halo needed for the horizontal window
    /// and anchor. Source-window and destination dimensions must match.
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
    /// The source window must match the destination size and retain the required halo.
    pub fn filter_row_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
    ) -> Result<()> {
        <Self as IntegerSeparableFilterImage<T, L>>::filter_row_image(
            stream_context,
            source,
            destination,
            kernel,
            anchor,
            divisor,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: IntegerSeparableFilterImage<T, L>,
{
    /// Applies a one-dimensional vertical filter with `i32` coefficients.
    /// Supply `kernel` in the reversed coefficient order required by NPP. `anchor` gives the
    /// zero-based kernel origin along the filtered axis. The result retains the image sample
    /// type. Passes the slice address directly as a device pointer, without uploading it;
    /// the coefficient storage must also be GPU-readable in the selected CUDA context.
    /// Divides the convolved sum by `divisor` before conversion to the output sample type.
    /// The wrapper forwards `divisor` without validation.
    /// Allocates a new owned image for the valid source window. Reduces height by
    /// `kernel.len() - 1` and preserves width.
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
    /// The source must contain the complete filter window and its valid output window.
    /// Image allocation or view construction, including alpha preservation copies, can also fail.
    pub fn filter_column(self, kernel: &[i32], anchor: i32, divisor: i32) -> Result<Self> {
        self.integer_separable_filter(
            FilterAxis::Column,
            kernel,
            anchor,
            divisor,
            <Self as IntegerSeparableFilterImage<T, L>>::filter_column_image,
        )
    }

    /// Applies a one-dimensional horizontal filter with `i32` coefficients.
    /// Supply `kernel` in the reversed coefficient order required by NPP. `anchor` gives the
    /// zero-based kernel origin along the filtered axis. The result retains the image sample
    /// type. Passes the slice address directly as a device pointer, without uploading it;
    /// the coefficient storage must also be GPU-readable in the selected CUDA context.
    /// Divides the convolved sum by `divisor` before conversion to the output sample type.
    /// The wrapper forwards `divisor` without validation.
    /// Allocates a new owned image for the valid source window. Reduces width by
    /// `kernel.len() - 1` and preserves height.
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
    /// The source must contain the complete filter window and its valid output window.
    /// Image allocation or view construction, including alpha preservation copies, can also fail.
    pub fn filter_row(self, kernel: &[i32], anchor: i32, divisor: i32) -> Result<Self> {
        self.integer_separable_filter(
            FilterAxis::Row,
            kernel,
            anchor,
            divisor,
            <Self as IntegerSeparableFilterImage<T, L>>::filter_row_image,
        )
    }

    fn integer_separable_filter(
        self,
        axis: FilterAxis,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
        filter: IntegerSeparableFilterCallback<L, T>,
    ) -> Result<Self> {
        self.ensure_eager()?;
        let (mask, origin) = axis.geometry(kernel.len(), anchor)?;
        let source = self.view()?.valid_window(mask, origin)?;
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
                anchor,
                divisor,
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

type IntegerSeparableFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageWindow<'_, T, L>,
    &mut ImageOutputView<'_, T, L>,
    &[i32],
    i32,
    i32,
) -> Result<()>;
