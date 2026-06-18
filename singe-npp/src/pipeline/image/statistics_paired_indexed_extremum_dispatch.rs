use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, MaskView},
    types::Point,
};

pub trait PairIndexedExtremumStatisticImage<T, L> {
    fn min_max_index(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        min: &mut DeviceMemory<T>,
        max: &mut DeviceMemory<T>,
        min_index: &mut DeviceMemory<Point>,
        max_index: &mut DeviceMemory<Point>,
    ) -> Result<()>;
}

pub trait ChannelPairIndexedExtremumStatisticImage<T, L> {
    fn min_max_index_channel(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        channel: usize,
        min: &mut DeviceMemory<T>,
        max: &mut DeviceMemory<T>,
        min_index: &mut DeviceMemory<Point>,
        max_index: &mut DeviceMemory<Point>,
    ) -> Result<()>;
}

pub trait MaskedPairIndexedExtremumStatisticImage<T, L> {
    fn min_max_index_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        min: &mut DeviceMemory<T>,
        max: &mut DeviceMemory<T>,
        min_index: &mut DeviceMemory<Point>,
        max_index: &mut DeviceMemory<Point>,
    ) -> Result<()>;
}

pub trait MaskedChannelPairIndexedExtremumStatisticImage<T, L> {
    fn min_max_index_channel_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        min: &mut DeviceMemory<T>,
        max: &mut DeviceMemory<T>,
        min_index: &mut DeviceMemory<Point>,
        max_index: &mut DeviceMemory<Point>,
    ) -> Result<()>;
}
