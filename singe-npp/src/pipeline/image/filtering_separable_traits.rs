use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageOutputView, ImageView, ImageWindow},
    types::{BorderType, Point},
};

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait IntegerSeparableFilterImage<T, L> {
    fn filter_column_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
    ) -> Result<()>;

    fn filter_row_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait FloatSeparableFilterImage<T, L> {
    fn filter_column32f_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f32],
        anchor: i32,
    ) -> Result<()>;

    fn filter_row32f_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f32],
        anchor: i32,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait DoubleSeparableFilterImage<T, L> {
    fn filter_column64f_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f64],
        anchor: i32,
    ) -> Result<()>;

    fn filter_row64f_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f64],
        anchor: i32,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait SumWindowFilterImage<T, L> {
    fn sum_window_column_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
    ) -> Result<()>;

    fn sum_window_row_image(
        stream_context: &StreamContext,
        source: &ImageWindow<'_, T, L>,
        destination: &mut ImageOutputView<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait IntegerSeparableBorderFilterImage<T, L> {
    fn filter_column_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_row_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[i32],
        anchor: i32,
        divisor: i32,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait FloatSeparableBorderFilterImage<T, L> {
    fn filter_column_border32f_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f32],
        anchor: i32,
        border_type: BorderType,
    ) -> Result<()>;

    fn filter_row_border32f_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, T, L>,
        kernel: &[f32],
        anchor: i32,
        border_type: BorderType,
    ) -> Result<()>;
}

/// # Safety
/// Preserve the native element/layout ABI and validate the complete initialized source footprint.
/// Complete borrowed native work on success, error, and unwind.
/// On success, initialize every output ROI sample except alpha for explicitly alpha-ignoring operations.
/// In-place operations preserve initialized values in every sample they leave untouched.
pub unsafe trait SumWindowBorderFilterImage<T, L> {
    fn sum_window_column_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
        border_type: BorderType,
    ) -> Result<()>;

    fn sum_window_row_border_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_offset: Point,
        destination: &mut ImageOutputView<'_, f32, L>,
        mask_size: i32,
        anchor: i32,
        border_type: BorderType,
    ) -> Result<()>;
}
