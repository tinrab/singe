use singe_cuda::view::{DeviceOutputView, DeviceRepr};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, MaskView},
    types::Point,
};

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait PairIndexedExtremumStatisticImage<T: DeviceRepr, L> {
    fn min_max_index(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        min: &mut DeviceOutputView<'_, T>,
        max: &mut DeviceOutputView<'_, T>,
        min_index: &mut DeviceOutputView<'_, Point>,
        max_index: &mut DeviceOutputView<'_, Point>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait ChannelPairIndexedExtremumStatisticImage<T: DeviceRepr, L> {
    fn min_max_index_channel(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        channel: usize,
        min: &mut DeviceOutputView<'_, T>,
        max: &mut DeviceOutputView<'_, T>,
        min_index: &mut DeviceOutputView<'_, Point>,
        max_index: &mut DeviceOutputView<'_, Point>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait MaskedPairIndexedExtremumStatisticImage<T: DeviceRepr, L> {
    fn min_max_index_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        min: &mut DeviceOutputView<'_, T>,
        max: &mut DeviceOutputView<'_, T>,
        min_index: &mut DeviceOutputView<'_, Point>,
        max_index: &mut DeviceOutputView<'_, Point>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait MaskedChannelPairIndexedExtremumStatisticImage<T: DeviceRepr, L> {
    fn min_max_index_channel_masked(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        min: &mut DeviceOutputView<'_, T>,
        max: &mut DeviceOutputView<'_, T>,
        min_index: &mut DeviceOutputView<'_, Point>,
        max_index: &mut DeviceOutputView<'_, Point>,
    ) -> Result<()>;
}
