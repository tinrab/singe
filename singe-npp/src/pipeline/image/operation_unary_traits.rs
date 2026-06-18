use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
};

pub trait AbsoluteImage<T, L> {
    fn absolute_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn absolute_image_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait AbsoluteDifferenceImage<T, L> {
    fn absolute_difference_image(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait AbsoluteDifferenceConstantImage<T> {
    fn absolute_difference_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        constant: T,
        destination: &mut ImageViewMut<'_, T, C1>,
    ) -> Result<()>;

    fn absolute_difference_device_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        constant: &DeviceMemory<T>,
        destination: &mut ImageViewMut<'_, T, C1>,
    ) -> Result<()>;
}

pub trait SquareImage<T, L> {
    fn square_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn square_image_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait SquareRootImage<T, L> {
    fn square_root_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn square_root_image_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait UnaryTranscendentalImage<T, L> {
    fn natural_logarithm_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn natural_logarithm_image_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn exponential_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn exponential_image_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait NaturalLogarithmImage<T, L> {
    fn natural_logarithm_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn natural_logarithm_image_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}
