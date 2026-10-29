use std::sync::Arc;

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
        statistics::{
            ImageIndexedMinMax, ImageIndexedStatistic, ImageStatistic,
            IndexedExtremumStatisticImage,
        },
    },
    types::Point,
};

mod paired_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: DeviceRepr,
    L: ChannelLayout,
    Self: IndexedExtremumStatisticImage<T, L>,
{
    /// Finds the minimum value and its source pixel coordinates in each channel.
    /// Coordinates are zero-based x/y positions relative to the supplied source view.
    /// The provided C1, C3, and C4 implementations produce one native `T` value and one
    /// `i32` x/y pair per channel; C4 includes the fourth channel.
    /// Each supplied output view must contain exactly the number of source channels.
    /// Prepares source and result regions, waits for producers on other streams,
    /// rejects capture, queries and allocates device scratch, and returns after successful
    /// stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native scratch-size queries or allocation, execution, or stream synchronization fail.
    /// All result views must have the exact element counts described above.
    pub fn min_index_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, T>,
        index_x: &mut DeviceOutputView<'_, i32>,
        index_y: &mut DeviceOutputView<'_, i32>,
    ) -> Result<()> {
        <Self as IndexedExtremumStatisticImage<T, L>>::min_index(
            stream_context,
            source,
            output,
            index_x,
            index_y,
        )
    }

    /// Finds the maximum value and its source pixel coordinates in each channel.
    /// Coordinates are zero-based x/y positions relative to the supplied source view.
    /// The provided C1, C3, and C4 implementations produce one native `T` value and one
    /// `i32` x/y pair per channel; C4 includes the fourth channel.
    /// Each supplied output view must contain exactly the number of source channels.
    /// Prepares source and result regions, waits for producers on other streams,
    /// rejects capture, queries and allocates device scratch, and returns after successful
    /// stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native scratch-size queries or allocation, execution, or stream synchronization fail.
    /// All result views must have the exact element counts described above.
    pub fn max_index_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, T>,
        index_x: &mut DeviceOutputView<'_, i32>,
        index_y: &mut DeviceOutputView<'_, i32>,
    ) -> Result<()> {
        <Self as IndexedExtremumStatisticImage<T, L>>::max_index(
            stream_context,
            source,
            output,
            index_x,
            index_y,
        )
    }

    /// Finds the minimum value and its source pixel coordinates in each channel.
    /// Coordinates are zero-based x/y positions relative to the supplied source view.
    /// The provided C1, C3, and C4 implementations produce one native `T` value and one
    /// `i32` x/y pair per channel; C4 includes the fourth channel.
    /// Allocates owned device arrays for the values and both coordinate components.
    /// Prepares source and result regions, waits for producers on other streams,
    /// rejects capture, queries and allocates device scratch, and returns after successful
    /// stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native scratch-size queries or allocation, execution, or stream synchronization fail.
    /// Result allocation or source view construction can also fail.
    pub fn min_index(self) -> Result<ImageIndexedStatistic<T>> {
        self.indexed_extremum_statistic(<Self as IndexedExtremumStatisticImage<T, L>>::min_index)
    }

    /// Finds the maximum value and its source pixel coordinates in each channel.
    /// Coordinates are zero-based x/y positions relative to the supplied source view.
    /// The provided C1, C3, and C4 implementations produce one native `T` value and one
    /// `i32` x/y pair per channel; C4 includes the fourth channel.
    /// Allocates owned device arrays for the values and both coordinate components.
    /// Prepares source and result regions, waits for producers on other streams,
    /// rejects capture, queries and allocates device scratch, and returns after successful
    /// stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native scratch-size queries or allocation, execution, or stream synchronization fail.
    /// Result allocation or source view construction can also fail.
    pub fn max_index(self) -> Result<ImageIndexedStatistic<T>> {
        self.indexed_extremum_statistic(<Self as IndexedExtremumStatisticImage<T, L>>::max_index)
    }

    fn indexed_extremum_statistic(
        self,
        statistic: IndexedExtremumStatisticCallback<L, T>,
    ) -> Result<ImageIndexedStatistic<T>> {
        self.ensure_eager()?;
        let output_channels = <Self as IndexedExtremumStatisticImage<T, L>>::OUTPUT_CHANNELS;
        let mut value =
            DeviceMemory::<T>::create_uninit(self.stream_context.cuda_context(), output_channels)?;
        let mut index_x = DeviceMemory::<i32>::create_uninit(
            self.stream_context.cuda_context(),
            output_channels,
        )?;
        let mut index_y = DeviceMemory::<i32>::create_uninit(
            self.stream_context.cuda_context(),
            output_channels,
        )?;
        {
            let source = self.view()?;
            statistic(
                self.stream_context,
                &source,
                &mut DeviceOutputView::from_output(&mut value),
                &mut DeviceOutputView::from_output(&mut index_x),
                &mut DeviceOutputView::from_output(&mut index_y),
            )?;
        }
        Ok(ImageIndexedStatistic {
            // SAFETY: the completed operation above wrote every element.
            value: ImageStatistic::from_values(unsafe { value.assume_init() }),
            // SAFETY: the completed operation above wrote every element.
            index_x: ImageStatistic::from_values(unsafe { index_x.assume_init() }),
            // SAFETY: the completed operation above wrote every element.
            index_y: ImageStatistic::from_values(unsafe { index_y.assume_init() }),
        })
    }
}

pub fn image_indexed_min_max<T>(
    min: DeviceMemory<T>,
    max: DeviceMemory<T>,
    min_index: DeviceMemory<Point>,
    max_index: DeviceMemory<Point>,
) -> Result<ImageIndexedMinMax<T>> {
    let min_index = split_points(min_index)?;
    let max_index = split_points(max_index)?;
    Ok(ImageIndexedMinMax {
        min: ImageIndexedStatistic {
            value: ImageStatistic::from_values(min),
            index_x: min_index.0,
            index_y: min_index.1,
        },
        max: ImageIndexedStatistic {
            value: ImageStatistic::from_values(max),
            index_x: max_index.0,
            index_y: max_index.1,
        },
    })
}

fn split_points(points: DeviceMemory<Point>) -> Result<(ImageStatistic<i32>, ImageStatistic<i32>)> {
    let host_points = points.copy_to_host_vec()?;
    let context = Arc::clone(points.context());
    drop(points);
    let mut index_x = Vec::with_capacity(host_points.len());
    let mut index_y = Vec::with_capacity(host_points.len());
    for point in host_points {
        index_x.push(point.x);
        index_y.push(point.y);
    }

    Ok((
        ImageStatistic::from_values(DeviceMemory::from_slice(&context, &index_x)?),
        ImageStatistic::from_values(DeviceMemory::from_slice(&context, &index_y)?),
    ))
}

type IndexedExtremumStatisticCallback<L, T> = fn(
    &StreamContext,
    &ImageView<'_, T, L>,
    &mut DeviceOutputView<'_, T>,
    &mut DeviceOutputView<'_, i32>,
    &mut DeviceOutputView<'_, i32>,
) -> Result<()>;
