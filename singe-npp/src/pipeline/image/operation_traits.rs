use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ImageView, ImageViewMut},
};

pub use super::operation_alpha_traits::*;
pub use super::operation_logical_traits::*;

#[path = "operation_binary_special_traits.rs"]
mod binary_special_traits;
#[path = "operation_constant_traits.rs"]
mod constant_traits;
#[path = "operation_unary_traits.rs"]
mod unary_traits;

pub use binary_special_traits::*;
pub use constant_traits::*;
pub use unary_traits::*;

pub trait BinaryArithmeticImage<T, L> {
    fn add_image(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn add_image_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn subtract_image(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn subtract_image_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn multiply_image(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn multiply_image_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn divide_image(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;

    fn divide_image_in_place(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        source_destination: &mut ImageViewMut<'_, T, L>,
        scale_factor: i32,
    ) -> Result<()>;
}

pub trait UnscaledBinaryArithmeticImage<T, L> {
    fn add_unscaled_image(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn subtract_unscaled_image(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn multiply_unscaled_image(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn divide_unscaled_image(
        stream_context: &StreamContext,
        left: &ImageView<'_, T, L>,
        right: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}
