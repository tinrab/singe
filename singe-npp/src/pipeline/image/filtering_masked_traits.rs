use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageOutputView, ImageView, ImageWindow},
    types::{BorderType, MaskSize, Point},
};

mod border_traits;

pub use border_traits::*;

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait MaskedKernelFilterImage<T, L> {
    fn filter_high_pass_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_low_pass_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_gauss_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_laplace_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait HighLowGaussFilterImage<T, L> {
    fn filter_high_pass_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_low_pass_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;

    fn filter_gauss_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        mask_size: MaskSize,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait AdvancedGaussFilterImage<T, L> {
    fn filter_gauss_advanced_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f32],
    ) -> Result<()>;

    fn filter_gauss_advanced_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f32],
        border_type: BorderType,
    ) -> Result<()>;
}
