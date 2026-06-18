use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
    types::Point,
};

use super::{
    ImagePipeline,
    statistics::{
        ImageIndexedMinMax, ImageIndexedStatistic, ImageStatistic, IndexedExtremumStatisticImage,
    },
};

#[path = "statistics_paired_indexed_extremum_methods.rs"]
mod paired_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: IndexedExtremumStatisticImage<T, L>,
{
    pub fn min_index_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<T>,
        index_x: &mut DeviceMemory<i32>,
        index_y: &mut DeviceMemory<i32>,
    ) -> Result<()> {
        <Self as IndexedExtremumStatisticImage<T, L>>::min_index(
            stream_context,
            source,
            output,
            index_x,
            index_y,
        )
    }

    pub fn max_index_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        output: &mut DeviceMemory<T>,
        index_x: &mut DeviceMemory<i32>,
        index_y: &mut DeviceMemory<i32>,
    ) -> Result<()> {
        <Self as IndexedExtremumStatisticImage<T, L>>::max_index(
            stream_context,
            source,
            output,
            index_x,
            index_y,
        )
    }

    pub fn min_index(self) -> Result<ImageIndexedStatistic<T>> {
        self.indexed_extremum_statistic(<Self as IndexedExtremumStatisticImage<T, L>>::min_index)
    }

    pub fn max_index(self) -> Result<ImageIndexedStatistic<T>> {
        self.indexed_extremum_statistic(<Self as IndexedExtremumStatisticImage<T, L>>::max_index)
    }

    fn indexed_extremum_statistic(
        self,
        statistic: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            &mut DeviceMemory<T>,
            &mut DeviceMemory<i32>,
            &mut DeviceMemory<i32>,
        ) -> Result<()>,
    ) -> Result<ImageIndexedStatistic<T>> {
        let output_channels = <Self as IndexedExtremumStatisticImage<T, L>>::OUTPUT_CHANNELS;
        let mut value = DeviceMemory::<T>::create(output_channels)?;
        let mut index_x = DeviceMemory::<i32>::create(output_channels)?;
        let mut index_y = DeviceMemory::<i32>::create(output_channels)?;
        {
            let source = self.view()?;
            statistic(
                self.stream_context,
                &source,
                &mut value,
                &mut index_x,
                &mut index_y,
            )?;
        }
        Ok(ImageIndexedStatistic {
            value: ImageStatistic::from_values(value),
            index_x: ImageStatistic::from_values(index_x),
            index_y: ImageStatistic::from_values(index_y),
        })
    }
}

pub(super) fn image_indexed_min_max<T>(
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
    let mut index_x = Vec::with_capacity(host_points.len());
    let mut index_y = Vec::with_capacity(host_points.len());
    for point in host_points {
        index_x.push(point.x);
        index_y.push(point.y);
    }

    Ok((
        ImageStatistic::from_values(DeviceMemory::from_slice(&index_x)?),
        ImageStatistic::from_values(DeviceMemory::from_slice(&index_y)?),
    ))
}
