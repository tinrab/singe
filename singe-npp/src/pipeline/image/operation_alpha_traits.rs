use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, ImageView, ImageViewMut},
    types::AlphaOperation,
};

pub trait AlphaPremultiplyConstantImage<T, L> {
    type Alpha;

    fn alpha_premultiply_constant_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        alpha: Self::Alpha,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;

    fn alpha_premultiply_constant_image_in_place(
        stream_context: &StreamContext,
        alpha: Self::Alpha,
        source_destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait ColorKeyImage<L> {
    type ColorKey;

    fn comp_color_key_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, u8, L>,
        source2: &ImageView<'_, u8, L>,
        color_key: Self::ColorKey,
        destination: &mut ImageViewMut<'_, u8, L>,
    ) -> Result<()>;
}

pub trait AlphaCompConstantImage<T, L> {
    type Alpha;

    fn alpha_comp_constant_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, L>,
        alpha1: Self::Alpha,
        source2: &ImageView<'_, T, L>,
        alpha2: Self::Alpha,
        destination: &mut ImageViewMut<'_, T, L>,
        operation: AlphaOperation,
    ) -> Result<()>;
}

pub trait AlphaCompImage<T, L> {
    fn alpha_comp_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, T, L>,
        source2: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
        operation: AlphaOperation,
    ) -> Result<()>;
}

pub trait AlphaPremultiplyImage<T> {
    fn alpha_premultiply_image(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, AC4>,
        destination: &mut ImageViewMut<'_, T, AC4>,
    ) -> Result<()>;

    fn alpha_premultiply_image_in_place(
        stream_context: &StreamContext,
        source_destination: &mut ImageViewMut<'_, T, AC4>,
    ) -> Result<()>;
}

pub trait AlphaCompColorKeyImage {
    fn alpha_comp_color_key_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, u8, AC4>,
        alpha1: u8,
        source2: &ImageView<'_, u8, AC4>,
        alpha2: u8,
        color_key: [u8; 4],
        destination: &mut ImageViewMut<'_, u8, AC4>,
        operation: AlphaOperation,
    ) -> Result<()>;
}
