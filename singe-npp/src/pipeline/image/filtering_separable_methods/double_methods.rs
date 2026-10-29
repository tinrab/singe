use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage,
        view::{ChannelLayout, ImageOutputView, ImageWindow},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline,
        filtering::{DoubleSeparableFilterImage, FilterAxis},
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: DoubleSeparableFilterImage<T, L>,
{
    /// Applies a one-dimensional vertical filter with `f64` coefficients.
    /// Supply `kernel` in the reversed coefficient order required by NPP. `anchor` gives the
    /// zero-based kernel origin along the filtered axis. The result retains the image sample
    /// type. Passes the slice address directly as a device pointer, without uploading it;
    /// the coefficient storage must also be GPU-readable in the selected CUDA context.
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
    pub fn filter_column64f_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f64],
        anchor: i32,
    ) -> Result<()> {
        <Self as DoubleSeparableFilterImage<T, L>>::filter_column64f_image(
            stream_context,
            source,
            destination,
            kernel,
            anchor,
        )
    }

    /// Applies a one-dimensional horizontal filter with `f64` coefficients.
    /// Supply `kernel` in the reversed coefficient order required by NPP. `anchor` gives the
    /// zero-based kernel origin along the filtered axis. The result retains the image sample
    /// type. Passes the slice address directly as a device pointer, without uploading it;
    /// the coefficient storage must also be GPU-readable in the selected CUDA context.
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
    pub fn filter_row64f_into(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f64],
        anchor: i32,
    ) -> Result<()> {
        <Self as DoubleSeparableFilterImage<T, L>>::filter_row64f_image(
            stream_context,
            source,
            destination,
            kernel,
            anchor,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy + SupportedImage<L>,
    L: ChannelLayout,
    Self: DoubleSeparableFilterImage<T, L>,
{
    /// Applies a one-dimensional vertical filter with `f64` coefficients.
    /// Supply `kernel` in the reversed coefficient order required by NPP. `anchor` gives the
    /// zero-based kernel origin along the filtered axis. The result retains the image sample
    /// type. Passes the slice address directly as a device pointer, without uploading it;
    /// the coefficient storage must also be GPU-readable in the selected CUDA context.
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
    pub fn filter_column64f(self, kernel: &[f64], anchor: i32) -> Result<Self> {
        self.double_separable_filter(
            FilterAxis::Column,
            kernel,
            anchor,
            <Self as DoubleSeparableFilterImage<T, L>>::filter_column64f_image,
        )
    }

    /// Applies a one-dimensional horizontal filter with `f64` coefficients.
    /// Supply `kernel` in the reversed coefficient order required by NPP. `anchor` gives the
    /// zero-based kernel origin along the filtered axis. The result retains the image sample
    /// type. Passes the slice address directly as a device pointer, without uploading it;
    /// the coefficient storage must also be GPU-readable in the selected CUDA context.
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
    pub fn filter_row64f(self, kernel: &[f64], anchor: i32) -> Result<Self> {
        self.double_separable_filter(
            FilterAxis::Row,
            kernel,
            anchor,
            <Self as DoubleSeparableFilterImage<T, L>>::filter_row64f_image,
        )
    }

    fn double_separable_filter(
        self,
        axis: FilterAxis,
        kernel: &[f64],
        anchor: i32,
        filter: DoubleSeparableFilterCallback<L, T>,
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

type DoubleSeparableFilterCallback<L, T> = fn(
    &StreamContext,
    &ImageWindow<'_, T, L>,
    &mut ImageOutputView<'_, T, L>,
    &[f64],
    i32,
) -> Result<()>;
