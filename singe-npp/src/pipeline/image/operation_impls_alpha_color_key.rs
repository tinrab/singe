use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::AlphaOperation,
};

use super::super::{
    ImagePipeline,
    operation_traits::{AlphaCompColorKeyImage, ColorKeyImage},
};

macro_rules! impl_color_key_image {
    ($layout:ty, $color_key:ty, $compose:path) => {
        impl<'a> ColorKeyImage<$layout> for ImagePipeline<'a, u8, $layout> {
            type ColorKey = $color_key;

            fn comp_color_key_image(
                stream_context: &StreamContext,
                source1: &ImageView<'_, u8, $layout>,
                source2: &ImageView<'_, u8, $layout>,
                color_key: Self::ColorKey,
                destination: &mut ImageViewMut<'_, u8, $layout>,
            ) -> Result<()> {
                $compose(stream_context, source1, source2, color_key, destination)
            }
        }
    };
}

impl_color_key_image!(C1, u8, arithmetic::comp_color_key_u8_c1);
impl_color_key_image!(C3, [u8; 3], arithmetic::comp_color_key_u8_c3);
impl_color_key_image!(C4, [u8; 4], arithmetic::comp_color_key_u8_c4);

impl<'a> AlphaCompColorKeyImage for ImagePipeline<'a, u8, AC4> {
    fn alpha_comp_color_key_image(
        stream_context: &StreamContext,
        source1: &ImageView<'_, u8, AC4>,
        alpha1: u8,
        source2: &ImageView<'_, u8, AC4>,
        alpha2: u8,
        color_key: [u8; 4],
        destination: &mut ImageViewMut<'_, u8, AC4>,
        operation: AlphaOperation,
    ) -> Result<()> {
        arithmetic::alpha_comp_color_key_u8_ac4(
            stream_context,
            source1,
            alpha1,
            source2,
            alpha2,
            color_key,
            destination,
            operation,
        )
    }
}
