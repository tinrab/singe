use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageOutputView, ImageView, ImageViewMut},
};

/// # Safety
/// Complete borrowed work on every path and write valid values to every active output channel.
/// Read-write operations require initialized inputs; AC4 dispatch preserves ignored alpha.
pub unsafe trait LogicalNotImage<T, L> {
    fn logical_not_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;

    fn logical_not_image_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed work on every path and write valid values to every active output channel.
/// Read-write operations require initialized inputs; AC4 dispatch preserves ignored alpha.
pub unsafe trait LogicalBinaryImage<T, L> {
    fn logical_and_image(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;

    fn logical_and_image_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn logical_or_image(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;

    fn logical_or_image_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn logical_xor_image(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;

    fn logical_xor_image_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed work on every path and write valid values to every active output channel.
/// Read-write operations require initialized inputs; AC4 dispatch preserves ignored alpha.
pub unsafe trait LogicalConstantImage<T, L> {
    type Constant;

    fn logical_and_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: Self::Constant,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;

    fn logical_and_constant_image_in_place(
        stream_context: &StreamContext,
        constant: Self::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn logical_or_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: Self::Constant,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;

    fn logical_or_constant_image_in_place(
        stream_context: &StreamContext,
        constant: Self::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn logical_xor_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: Self::Constant,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;

    fn logical_xor_constant_image_in_place(
        stream_context: &StreamContext,
        constant: Self::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed work on every path and write valid values to every active output channel.
/// Read-write operations require initialized inputs; AC4 dispatch preserves ignored alpha.
pub unsafe trait RightShiftConstantImage<T, L> {
    type Constant;

    fn right_shift_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: Self::Constant,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;

    fn right_shift_constant_image_in_place(
        stream_context: &StreamContext,
        constant: Self::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed work on every path and write valid values to every active output channel.
/// Read-write operations require initialized inputs; AC4 dispatch preserves ignored alpha.
pub unsafe trait LeftShiftConstantImage<T, L> {
    type Constant;

    fn left_shift_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: Self::Constant,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;

    fn left_shift_constant_image_in_place(
        stream_context: &StreamContext,
        constant: Self::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}
