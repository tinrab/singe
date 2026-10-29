use singe_cuda::{
    memory::DeviceMemory,
    view::{DeviceOutputView, DeviceRepr},
};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
    pipeline::image::{
        ImagePipeline,
        statistics::{ExtremumStatisticImage, ImageMinMax, ImageStatistic},
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: DeviceRepr,
    L: ChannelLayout,
    Self: ExtremumStatisticImage<T, L>,
{
    /// Computes native minimum samples independently for each channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order,
    /// using the source element type.
    /// Each output view must contain exactly one element per processed channel.
    /// The native reduction owns its temporary scratch.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Each output length must equal the dispatch channel count.
    /// Scratch-size queries and allocation can also fail.
    pub fn min_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, T>,
    ) -> Result<()> {
        <Self as ExtremumStatisticImage<T, L>>::min(stream_context, source, output)
    }

    /// Computes native maximum samples independently for each channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order,
    /// using the source element type.
    /// Each output view must contain exactly one element per processed channel.
    /// The native reduction owns its temporary scratch.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Each output length must equal the dispatch channel count.
    /// Scratch-size queries and allocation can also fail.
    pub fn max_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, T>,
    ) -> Result<()> {
        <Self as ExtremumStatisticImage<T, L>>::max(stream_context, source, output)
    }

    /// Computes native minimum and maximum samples independently for each channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order,
    /// using the source element type.
    /// Each output view must contain exactly one element per processed channel.
    /// The native reduction owns its temporary scratch.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Each output length must equal the dispatch channel count.
    /// Scratch-size queries and allocation can also fail.
    pub fn min_max_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        min: &mut DeviceOutputView<'_, T>,
        max: &mut DeviceOutputView<'_, T>,
    ) -> Result<()> {
        <Self as ExtremumStatisticImage<T, L>>::min_max(stream_context, source, min, max)
    }

    /// Computes native minimum samples independently for each channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order,
    /// using the source element type.
    /// Allocates owned device result storage.
    /// The native reduction owns its temporary scratch.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Allocating result buffers and constructing the source view can also fail.
    /// Scratch-size queries and allocation can also fail.
    pub fn min(self) -> Result<ImageStatistic<T>> {
        self.extremum_statistic(<Self as ExtremumStatisticImage<T, L>>::min)
    }

    /// Computes native maximum samples independently for each channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order,
    /// using the source element type.
    /// Allocates owned device result storage.
    /// The native reduction owns its temporary scratch.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Allocating result buffers and constructing the source view can also fail.
    /// Scratch-size queries and allocation can also fail.
    pub fn max(self) -> Result<ImageStatistic<T>> {
        self.extremum_statistic(<Self as ExtremumStatisticImage<T, L>>::max)
    }

    /// Computes native minimum and maximum samples independently for each channel.
    /// The provided C1, C3, and C4 implementations return one result per channel in channel order,
    /// using the source element type.
    /// Allocates owned device result storage for each of the two results.
    /// The native reduction owns its temporary scratch.
    /// Prepares participating regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Allocating result buffers and constructing the source view can also fail.
    /// Scratch-size queries and allocation can also fail.
    pub fn min_max(self) -> Result<ImageMinMax<T>> {
        self.ensure_eager()?;
        let mut min = DeviceMemory::<T>::create_uninit(
            self.stream_context.cuda_context(),
            <Self as ExtremumStatisticImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        let mut max = DeviceMemory::<T>::create_uninit(
            self.stream_context.cuda_context(),
            <Self as ExtremumStatisticImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            <Self as ExtremumStatisticImage<T, L>>::min_max(
                self.stream_context,
                &source,
                &mut DeviceOutputView::from_output(&mut min),
                &mut DeviceOutputView::from_output(&mut max),
            )?;
        }
        Ok(ImageMinMax {
            // SAFETY: the completed operation above wrote every element.
            min: ImageStatistic::from_values(unsafe { min.assume_init() }),
            // SAFETY: the completed operation above wrote every element.
            max: ImageStatistic::from_values(unsafe { max.assume_init() }),
        })
    }

    fn extremum_statistic(
        self,
        statistic: ExtremumStatisticCallback<L, T>,
    ) -> Result<ImageStatistic<T>> {
        self.ensure_eager()?;
        let mut values = DeviceMemory::<T>::create_uninit(
            self.stream_context.cuda_context(),
            <Self as ExtremumStatisticImage<T, L>>::OUTPUT_CHANNELS,
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

type ExtremumStatisticCallback<L, T> =
    fn(&StreamContext, &ImageView<'_, T, L>, &mut DeviceOutputView<'_, T>) -> Result<()>;
