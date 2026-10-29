use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageOutputView, ImageView, ImageViewMut},
};

mod device_constant_traits;

pub use device_constant_traits::{DeviceConstantArithmeticImage, MultiplyDeviceConstantScaleImage};

/// # Safety
/// Complete borrowed work on every path and write valid values to every active output channel.
/// Read-write operations require initialized inputs; AC4 dispatch preserves ignored alpha.
pub unsafe trait ConstantArithmeticImage<T, L> {
    type Constant;

    fn add_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: Self::Constant,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn add_constant_image_in_place(
        stream_context: &StreamContext,
        constant: Self::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn subtract_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: Self::Constant,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn subtract_constant_image_in_place(
        stream_context: &StreamContext,
        constant: Self::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn multiply_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: Self::Constant,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn multiply_constant_image_in_place(
        stream_context: &StreamContext,
        constant: Self::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn divide_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: Self::Constant,
        destination: &mut ImageOutputView<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn divide_constant_image_in_place(
        stream_context: &StreamContext,
        constant: Self::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;
}

/// # Safety
/// Complete borrowed work on every path and write valid values to every active output channel.
/// Read-write operations require initialized inputs; AC4 dispatch preserves ignored alpha.
pub unsafe trait MultiplyConstantScaleImage<T, L> {
    type Constant;

    fn multiply_constant_scale_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: Self::Constant,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>;

    fn multiply_constant_scale_image_in_place(
        stream_context: &StreamContext,
        constant: Self::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}
