use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
    pipeline::image::{
        ImagePipeline,
        statistics::{ErrorMetricImage, ImageStatistic},
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ErrorMetricImage<T, L>,
{
    /// Computes the native maximum error between two image ROIs.
    /// The provided C1, C3, and C4 implementations produce one aggregate `f64` result
    /// across the image channels, including the fourth channel in C4.
    /// The supplied output view must contain exactly one element.
    /// Prepares both source images and output storage, waits for producers on other
    /// streams, rejects capture, queries and allocates temporary device scratch, and returns
    /// after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native execution, or stream synchronization fail.
    /// Source image dimensions must match.
    /// The output view must have the exact result count described above.
    pub fn maximum_error_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as ErrorMetricImage<T, L>>::maximum_error(stream_context, source_1, source_2, output)
    }

    /// Computes the native average error between two image ROIs.
    /// The provided C1, C3, and C4 implementations produce one aggregate `f64` result
    /// across the image channels, including the fourth channel in C4.
    /// The supplied output view must contain exactly one element.
    /// Prepares both source images and output storage, waits for producers on other
    /// streams, rejects capture, queries and allocates temporary device scratch, and returns
    /// after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native execution, or stream synchronization fail.
    /// Source image dimensions must match.
    /// The output view must have the exact result count described above.
    pub fn average_error_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as ErrorMetricImage<T, L>>::average_error(stream_context, source_1, source_2, output)
    }

    /// Computes the native maximum error between two image ROIs.
    /// The provided C1, C3, and C4 implementations produce one aggregate `f64` result
    /// across the image channels, including the fourth channel in C4.
    /// Allocates owned device storage for the single result.
    /// Prepares both source images and output storage, waits for producers on other
    /// streams, rejects capture, queries and allocates temporary device scratch, and returns
    /// after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native execution, or stream synchronization fail.
    /// Source image dimensions must match.
    /// Result allocation or source view construction can also fail.
    pub fn maximum_error(self, other: &ImageView<'_, T, L>) -> Result<ImageStatistic<f64>> {
        self.error_metric(other, <Self as ErrorMetricImage<T, L>>::maximum_error)
    }

    /// Computes the native average error between two image ROIs.
    /// The provided C1, C3, and C4 implementations produce one aggregate `f64` result
    /// across the image channels, including the fourth channel in C4.
    /// Allocates owned device storage for the single result.
    /// Prepares both source images and output storage, waits for producers on other
    /// streams, rejects capture, queries and allocates temporary device scratch, and returns
    /// after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native execution, or stream synchronization fail.
    /// Source image dimensions must match.
    /// Result allocation or source view construction can also fail.
    pub fn average_error(self, other: &ImageView<'_, T, L>) -> Result<ImageStatistic<f64>> {
        self.error_metric(other, <Self as ErrorMetricImage<T, L>>::average_error)
    }

    fn error_metric(
        self,
        other: &ImageView<'_, T, L>,
        metric: ErrorMetricCallback<L, T>,
    ) -> Result<ImageStatistic<f64>> {
        self.ensure_eager()?;
        let mut output = DeviceMemory::<f64>::create_uninit(self.stream_context.cuda_context(), 1)?;
        {
            let source = self.view()?;
            metric(
                self.stream_context,
                &source,
                other,
                &mut DeviceOutputView::from_output(&mut output),
            )?;
        }
        // SAFETY: the completed operation above wrote every element.
        Ok(ImageStatistic::from_values(unsafe { output.assume_init() }))
    }
}

type ErrorMetricCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    &ImageView<'_, T, L>,
    &mut DeviceOutputView<'_, f64>,
) -> Result<()>;
