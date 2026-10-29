use singe_cuda::view::DeviceOutputView;

pub use self::relative_norm::{MaskedChannelNormRelativeImage, MaskedNormRelativeImage};
use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, MaskView},
};

mod relative_norm;

/// # Safety
/// Complete borrowed native work on every path and uphold the declared operation coverage and ABI.
/// On success, initialize every element of each supplied output view.
pub unsafe trait MaskedNormDiffImage<T, L> {
    fn norm_diff_inf_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    fn norm_diff_l1_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    fn norm_diff_l2_masked(
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
pub unsafe trait MaskedChannelNormDiffImage<T, L> {
    fn norm_diff_inf_channel_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    fn norm_diff_l1_channel_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;

    fn norm_diff_l2_channel_masked(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        channel: usize,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()>;
}

macro_rules! impl_masked_norm_diff_image {
    ($ty:ty, $norm_inf:path, $norm_l1:path, $norm_l2:path) => {
        unsafe impl<'a> MaskedNormDiffImage<$ty, C1> for ImagePipeline<'a, $ty, C1> {
            fn norm_diff_inf_masked(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, C1>,
                source_2: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_1)?;
                stream_context.prepare_region(source_2)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $norm_inf(stream_context, source_1, source_2, mask, output) }
            }

            fn norm_diff_l1_masked(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, C1>,
                source_2: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_1)?;
                stream_context.prepare_region(source_2)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $norm_l1(stream_context, source_1, source_2, mask, output) }
            }

            fn norm_diff_l2_masked(
                stream_context: &StreamContext,
                source_1: &ImageView<'_, $ty, C1>,
                source_2: &ImageView<'_, $ty, C1>,
                mask: &MaskView<'_>,
                output: &mut DeviceOutputView<'_, f64>,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_1)?;
                stream_context.prepare_region(source_2)?;
                stream_context.prepare_region(mask)?;
                stream_context.prepare_region(output)?;
                // SAFETY: the regions were prepared for this stream above, so earlier producers are ordered before this launch.
                unsafe { $norm_l2(stream_context, source_1, source_2, mask, output) }
            }
        }
    };
}

#[macro_use]
mod channel_norm_dispatch_macros;

#[macro_use]
mod relative_norm_dispatch_macros;
