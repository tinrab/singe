use singe_cuda::view::DeviceOutputView;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, MaskView},
};

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait MaskedNormRelativeImage<T, L> {
    fn norm_relative_inf_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    fn norm_relative_l1_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    fn norm_relative_l2_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait MaskedChannelNormRelativeImage<T, L> {
    fn norm_relative_inf_channel_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    fn norm_relative_l1_channel_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    fn norm_relative_l2_channel_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;
}
