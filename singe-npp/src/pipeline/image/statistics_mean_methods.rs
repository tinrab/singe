use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
    pipeline::image::{
        ImagePipeline,
        statistics::{
            ChannelMeanStandardDeviationImage, ImageMeanStandardDeviation, ImageStatistic,
            MeanStandardDeviationImage,
        },
    },
};

mod masked_channel_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MeanStandardDeviationImage<T, L>,
{
    /// Computes the native mean and standard deviation of the C1 image.
    /// The provided implementation processes C1.
    /// Each supplied result view must contain exactly one `f64` element.
    /// Queries and allocates temporary device scratch for the native reduction.
    /// Prepares participating image and result regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Scratch-size queries and device scratch allocation can also fail.
    /// Each result view must contain exactly one element.
    pub fn mean_standard_deviation_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mean: &mut DeviceOutputView<'_, f64>,
        standard_deviation: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MeanStandardDeviationImage<T, L>>::mean_standard_deviation(
            stream_context,
            source,
            mean,
            standard_deviation,
        )
    }

    /// Computes the native mean and standard deviation of the C1 image.
    /// The provided implementation processes C1.
    /// Allocates owned device storage containing one `f64` mean and one `f64` standard deviation.
    /// Queries and allocates temporary device scratch for the native reduction.
    /// Prepares participating image and result regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Scratch-size queries and device scratch allocation can also fail.
    /// Result allocation or source view construction can also fail.
    pub fn mean_standard_deviation(self) -> Result<ImageMeanStandardDeviation<f64>> {
        self.ensure_eager()?;
        let mut mean = DeviceMemory::<f64>::create_uninit(
            self.stream_context.cuda_context(),
            <Self as MeanStandardDeviationImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        let mut standard_deviation = DeviceMemory::<f64>::create_uninit(
            self.stream_context.cuda_context(),
            <Self as MeanStandardDeviationImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            <Self as MeanStandardDeviationImage<T, L>>::mean_standard_deviation(
                self.stream_context,
                &source,
                &mut DeviceOutputView::from_output(&mut mean),
                &mut DeviceOutputView::from_output(&mut standard_deviation),
            )?;
        }
        Ok(ImageMeanStandardDeviation {
            // SAFETY: the completed operation above wrote every element.
            mean: ImageStatistic::from_values(unsafe { mean.assume_init() }),
            // SAFETY: the completed operation above wrote every element.
            standard_deviation: ImageStatistic::from_values(unsafe {
                standard_deviation.assume_init()
            }),
        })
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ChannelMeanStandardDeviationImage<T, L>,
{
    /// Computes the native mean and standard deviation of one selected channel.
    /// The provided implementation processes C3 with zero-based `channel` in `0..3`.
    /// Each supplied result view must contain exactly one `f64` element.
    /// Queries and allocates temporary device scratch for the native reduction.
    /// Prepares participating image and result regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Scratch-size queries and device scratch allocation can also fail.
    /// Each result view must contain exactly one element.
    /// `channel` must be less than three.
    pub fn mean_standard_deviation_channel_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        channel: usize,
        mean: &mut DeviceOutputView<'_, f64>,
        standard_deviation: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as ChannelMeanStandardDeviationImage<T, L>>::mean_standard_deviation_channel(
            stream_context,
            source,
            channel,
            mean,
            standard_deviation,
        )
    }

    /// Computes the native mean and standard deviation of one selected channel.
    /// The provided implementation processes C3 with zero-based `channel` in `0..3`.
    /// Allocates owned device storage containing one `f64` mean and one `f64` standard deviation.
    /// Queries and allocates temporary device scratch for the native reduction.
    /// Prepares participating image and result regions, waits for producers on other streams,
    /// rejects capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Scratch-size queries and device scratch allocation can also fail.
    /// Result allocation or source view construction can also fail.
    /// `channel` must be less than three.
    pub fn mean_standard_deviation_channel(
        self,
        channel: usize,
    ) -> Result<ImageMeanStandardDeviation<f64>> {
        self.ensure_eager()?;
        let mut mean = DeviceMemory::<f64>::create_uninit(self.stream_context.cuda_context(), 1)?;
        let mut standard_deviation =
            DeviceMemory::<f64>::create_uninit(self.stream_context.cuda_context(), 1)?;
        {
            let source = self.view()?;
            <Self as ChannelMeanStandardDeviationImage<T, L>>::mean_standard_deviation_channel(
                self.stream_context,
                &source,
                channel,
                &mut DeviceOutputView::from_output(&mut mean),
                &mut DeviceOutputView::from_output(&mut standard_deviation),
            )?;
        }
        Ok(ImageMeanStandardDeviation {
            // SAFETY: the completed operation above wrote every element.
            mean: ImageStatistic::from_values(unsafe { mean.assume_init() }),
            // SAFETY: the completed operation above wrote every element.
            standard_deviation: ImageStatistic::from_values(unsafe {
                standard_deviation.assume_init()
            }),
        })
    }
}
