use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
};

#[path = "operation_device_constant_traits.rs"]
mod device_constant_traits;

pub use device_constant_traits::{DeviceConstantArithmeticImage, MultiplyDeviceConstantScaleImage};

pub trait ConstantArithmeticImage<T, L> {
    type Constant;

    fn add_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: Self::Constant,
        destination: &mut ImageViewMut<'_, T, L>,
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
        destination: &mut ImageViewMut<'_, T, L>,
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
        destination: &mut ImageViewMut<'_, T, L>,
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
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn divide_constant_image_in_place(
        stream_context: &StreamContext,
        constant: Self::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait MultiplyConstantScaleImage<T, L> {
    type Constant;

    fn multiply_constant_scale_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        constant: Self::Constant,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn multiply_constant_scale_image_in_place(
        stream_context: &StreamContext,
        constant: Self::Constant,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}
