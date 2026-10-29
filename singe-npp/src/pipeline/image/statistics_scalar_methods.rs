use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
    pipeline::image::{
        ImagePipeline,
        statistics::{ImageStatistic, ScalarStatisticImage},
    },
};

mod masked_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ScalarStatisticImage<T, L>,
{
    /// Computes the sum of samples for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Writes `f64` results into `output`, whose length must exactly match the dispatch output count.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The output length must equal the required number of channel results.
    pub fn sum_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as ScalarStatisticImage<T, L>>::sum(stream_context, source, output)
    }

    /// Computes the arithmetic mean of samples for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Writes `f64` results into `output`, whose length must exactly match the dispatch output count.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The output length must equal the required number of channel results.
    pub fn mean_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as ScalarStatisticImage<T, L>>::mean(stream_context, source, output)
    }

    /// Computes the maximum absolute sample value for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Writes `f64` results into `output`, whose length must exactly match the dispatch output count.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The output length must equal the required number of channel results.
    pub fn norm_inf_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as ScalarStatisticImage<T, L>>::norm_inf(stream_context, source, output)
    }

    /// Computes the sum of absolute sample values for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Writes `f64` results into `output`, whose length must exactly match the dispatch output count.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The output length must equal the required number of channel results.
    pub fn norm_l1_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as ScalarStatisticImage<T, L>>::norm_l1(stream_context, source, output)
    }

    /// Computes the square root of the sum of squared sample values for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Writes `f64` results into `output`, whose length must exactly match the dispatch output count.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// The output length must equal the required number of channel results.
    pub fn norm_l2_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as ScalarStatisticImage<T, L>>::norm_l2(stream_context, source, output)
    }

    /// Computes the sum of samples for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Allocates owned device result storage and returns it as an `ImageStatistic<f64>`.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// Allocating result storage or constructing the source view can also fail.
    pub fn sum(self) -> Result<ImageStatistic<f64>> {
        self.scalar_statistic(<Self as ScalarStatisticImage<T, L>>::sum)
    }

    /// Computes the arithmetic mean of samples for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Allocates owned device result storage and returns it as an `ImageStatistic<f64>`.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// Allocating result storage or constructing the source view can also fail.
    pub fn mean(self) -> Result<ImageStatistic<f64>> {
        self.scalar_statistic(<Self as ScalarStatisticImage<T, L>>::mean)
    }

    /// Computes the maximum absolute sample value for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Allocates owned device result storage and returns it as an `ImageStatistic<f64>`.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// Allocating result storage or constructing the source view can also fail.
    pub fn norm_inf(self) -> Result<ImageStatistic<f64>> {
        self.scalar_statistic(<Self as ScalarStatisticImage<T, L>>::norm_inf)
    }

    /// Computes the sum of absolute sample values for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Allocates owned device result storage and returns it as an `ImageStatistic<f64>`.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// Allocating result storage or constructing the source view can also fail.
    pub fn norm_l1(self) -> Result<ImageStatistic<f64>> {
        self.scalar_statistic(<Self as ScalarStatisticImage<T, L>>::norm_l1)
    }

    /// Computes the square root of the sum of squared sample values for each processed channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order.
    /// Allocates owned device result storage and returns it as an `ImageStatistic<f64>`.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful native completion. The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native reduction, or stream synchronization fail.
    /// Allocating result storage or constructing the source view can also fail.
    pub fn norm_l2(self) -> Result<ImageStatistic<f64>> {
        self.scalar_statistic(<Self as ScalarStatisticImage<T, L>>::norm_l2)
    }

    fn scalar_statistic(
        self,
        statistic: ScalarStatisticCallback<L, T>,
    ) -> Result<ImageStatistic<f64>> {
        self.ensure_eager()?;
        let mut values = DeviceMemory::<f64>::create_uninit(
            self.stream_context.cuda_context(),
            <Self as ScalarStatisticImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            statistic(
                self.stream_context,
                &source,
                &mut DeviceOutputView::from_output(&mut values),
            )?;
        }
        // SAFETY: the completed operation above wrote every element.
        Ok(ImageStatistic::from_values(unsafe { values.assume_init() }))
    }
}

type ScalarStatisticCallback<L, T> =
    fn(&StreamContext, &ImageView<'_, T, L>, &mut DeviceOutputView<'_, f64>) -> Result<()>;
