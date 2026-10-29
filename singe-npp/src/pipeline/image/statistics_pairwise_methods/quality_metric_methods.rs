use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
    pipeline::image::{
        ImagePipeline,
        statistics::{ImageStatistic, QualityMetricImage},
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: QualityMetricImage<T, L>,
{
    /// Computes native mean squared error (MSE) between two image ROIs.
    /// The provided C1 and C3 implementations produce one or three `f32` results respectively.
    /// The supplied output view must contain exactly the result count described above.
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
    pub fn mean_squared_error_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f32>,
    ) -> Result<()> {
        <Self as QualityMetricImage<T, L>>::mean_squared_error(
            stream_context,
            source_1,
            source_2,
            output,
        )
    }

    /// Computes native peak signal-to-noise ratio (PSNR) between two image ROIs.
    /// The provided C1 and C3 implementations produce one or three `f32` results respectively.
    /// The supplied output view must contain exactly the result count described above.
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
    pub fn peak_signal_to_noise_ratio_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f32>,
    ) -> Result<()> {
        <Self as QualityMetricImage<T, L>>::peak_signal_to_noise_ratio(
            stream_context,
            source_1,
            source_2,
            output,
        )
    }

    /// Computes native mean squared error (MSE) between two image ROIs.
    /// The provided C1 and C3 implementations produce one or three `f32` results respectively.
    /// Allocates owned device storage for the complete result array.
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
    pub fn mean_squared_error(self, other: &ImageView<'_, T, L>) -> Result<ImageStatistic<f32>> {
        self.quality_metric(
            other,
            <Self as QualityMetricImage<T, L>>::mean_squared_error,
        )
    }

    /// Computes native peak signal-to-noise ratio (PSNR) between two image ROIs.
    /// The provided C1 and C3 implementations produce one or three `f32` results respectively.
    /// Allocates owned device storage for the complete result array.
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
    pub fn peak_signal_to_noise_ratio(
        self,
        other: &ImageView<'_, T, L>,
    ) -> Result<ImageStatistic<f32>> {
        self.quality_metric(
            other,
            <Self as QualityMetricImage<T, L>>::peak_signal_to_noise_ratio,
        )
    }

    fn quality_metric(
        self,
        other: &ImageView<'_, T, L>,
        metric: QualityMetricCallback<L, T>,
    ) -> Result<ImageStatistic<f32>> {
        self.ensure_eager()?;
        let mut output = DeviceMemory::<f32>::create_uninit(
            self.stream_context.cuda_context(),
            <Self as QualityMetricImage<T, L>>::OUTPUT_CHANNELS,
        )?;
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

type QualityMetricCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    &ImageView<'_, T, L>,
    &mut DeviceOutputView<'_, f32>,
) -> Result<()>;
