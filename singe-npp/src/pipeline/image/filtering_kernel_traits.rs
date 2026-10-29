use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageOutputView, ImageView, ImageWindow},
    types::{BorderType, Point, Size},
};

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait IntegerKernelFilterImage<T, L> {
    fn filter_kernel_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[i32],
        kernel_size: Size,
        anchor: Point,
        divisor: i32,
    ) -> Result<()>;

    fn filter_kernel_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[i32],
        kernel_size: Size,
        anchor: Point,
        divisor: i32,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait FloatKernelFilterImage<T, L> {
    fn filter_kernel32f_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<()>;

    fn filter_kernel32f_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait DoubleKernelFilterImage<T, L> {
    fn filter_kernel64f_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f64],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait TypedFloatKernelFilterImage<S, SL, D, DL> {
    fn filter_kernel32f_to_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, S, SL>,
        destination: &mut ImageOutputView<'_, D, DL>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
    ) -> Result<()>;

    fn filter_kernel32f_border_to_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, S, SL>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, D, DL>,
        kernel: &[f32],
        kernel_size: Size,
        anchor: Point,
        border_type: BorderType,
    ) -> Result<()>;
}
