use std::marker::PhantomData;

use singe_cuda::memory::DeviceMemory;

use crate::{
    error::Result,
    image::view::{C1, ChannelLayout, ImageView, ImageViewMut},
    types::Size,
    utility::checked_len,
};

#[derive(Debug)]
pub struct ImageStatistic<T> {
    values: DeviceMemory<T>,
}

impl<T> ImageStatistic<T> {
    pub(in crate::pipeline::image) const fn from_values(values: DeviceMemory<T>) -> Self {
        Self { values }
    }

    pub const fn len(&self) -> usize {
        self.values.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn values(&self) -> &DeviceMemory<T> {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut DeviceMemory<T> {
        &mut self.values
    }

    pub fn into_values(self) -> DeviceMemory<T> {
        self.values
    }
}

#[derive(Debug)]
pub struct ImageHistograms {
    pub channels: Vec<ImageStatistic<i32>>,
}

impl ImageHistograms {
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }
}

#[derive(Debug)]
pub struct ImageMeanStandardDeviation<T> {
    pub mean: ImageStatistic<T>,
    pub standard_deviation: ImageStatistic<T>,
}

#[derive(Debug)]
pub struct ImageMinMax<T> {
    pub min: ImageStatistic<T>,
    pub max: ImageStatistic<T>,
}

#[derive(Debug)]
pub struct ImageIndexedStatistic<T> {
    pub value: ImageStatistic<T>,
    pub index_x: ImageStatistic<i32>,
    pub index_y: ImageStatistic<i32>,
}

#[derive(Debug)]
pub struct ImageIndexedMinMax<T> {
    pub min: ImageIndexedStatistic<T>,
    pub max: ImageIndexedStatistic<T>,
}

#[derive(Debug)]
pub struct ContiguousImage<T, L = C1> {
    values: DeviceMemory<T>,
    size: Size,
    _marker: PhantomData<L>,
}

impl<T, L> ContiguousImage<T, L>
where
    L: ChannelLayout,
{
    pub fn create(size: Size) -> Result<Self> {
        let len = checked_len(size, L::CHANNELS)?;
        Ok(Self {
            values: DeviceMemory::<T>::create(len)?,
            size,
            _marker: PhantomData,
        })
    }

    pub const fn size(&self) -> Size {
        self.size
    }

    pub const fn values(&self) -> &DeviceMemory<T> {
        &self.values
    }

    pub fn into_values(self) -> DeviceMemory<T> {
        self.values
    }

    pub fn view(&self) -> Result<ImageView<'_, T, L>> {
        ImageView::from_memory(&self.values, self.size)
    }

    pub fn view_mut(&mut self) -> Result<ImageViewMut<'_, T, L>> {
        ImageViewMut::from_memory(&mut self.values, self.size)
    }
}

#[derive(Debug)]
pub struct ImageSquaredIntegral<T, S> {
    pub integral: ContiguousImage<T, C1>,
    pub squared: ContiguousImage<S, C1>,
}
