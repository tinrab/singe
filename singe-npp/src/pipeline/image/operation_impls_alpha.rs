use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::{ImagePipeline, operation_traits::*};

macro_rules! impl_alpha_premultiply_constant_image {
    ($ty:ty, $layout:ty, $alpha:ty, $premultiply:path, $premultiply_in_place:path) => {
        impl<'a> AlphaPremultiplyConstantImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Alpha = $alpha;

            fn alpha_premultiply_constant_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                alpha: Self::Alpha,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $premultiply(stream_context, source, alpha, destination)
            }

            fn alpha_premultiply_constant_image_in_place(
                stream_context: &StreamContext,
                alpha: Self::Alpha,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $premultiply_in_place(stream_context, alpha, source_destination)
            }
        }
    };
}

macro_rules! impl_alpha_comp_constant_image {
    ($ty:ty, $layout:ty, $alpha:ty, $compose:path) => {
        impl<'a> AlphaCompConstantImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            type Alpha = $alpha;

            fn alpha_comp_constant_image(
                stream_context: &StreamContext,
                source1: &ImageView<'_, $ty, $layout>,
                alpha1: Self::Alpha,
                source2: &ImageView<'_, $ty, $layout>,
                alpha2: Self::Alpha,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                operation: AlphaOperation,
            ) -> Result<()> {
                $compose(
                    stream_context,
                    source1,
                    alpha1,
                    source2,
                    alpha2,
                    destination,
                    operation,
                )
            }
        }
    };
}

macro_rules! impl_alpha_comp_image {
    ($ty:ty, $layout:ty, $compose:path) => {
        impl<'a> AlphaCompImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn alpha_comp_image(
                stream_context: &StreamContext,
                source1: &ImageView<'_, $ty, $layout>,
                source2: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                operation: AlphaOperation,
            ) -> Result<()> {
                $compose(stream_context, source1, source2, destination, operation)
            }
        }
    };
}

macro_rules! impl_alpha_premultiply_image {
    ($ty:ty, $premultiply:path, $premultiply_in_place:path) => {
        impl<'a> AlphaPremultiplyImage<$ty> for ImagePipeline<'a, $ty, AC4> {
            fn alpha_premultiply_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, AC4>,
                destination: &mut ImageViewMut<'_, $ty, AC4>,
            ) -> Result<()> {
                $premultiply(stream_context, source, destination)
            }

            fn alpha_premultiply_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, AC4>,
            ) -> Result<()> {
                $premultiply_in_place(stream_context, source_destination)
            }
        }
    };
}

impl_alpha_premultiply_constant_image!(
    u8,
    C1,
    u8,
    arithmetic::alpha_premultiply_constant_u8_c1,
    arithmetic::alpha_premultiply_constant_u8_c1_in_place
);
impl_alpha_premultiply_constant_image!(
    u8,
    C3,
    u8,
    arithmetic::alpha_premultiply_constant_u8_c3,
    arithmetic::alpha_premultiply_constant_u8_c3_in_place
);
impl_alpha_premultiply_constant_image!(
    u8,
    C4,
    u8,
    arithmetic::alpha_premultiply_constant_u8_c4,
    arithmetic::alpha_premultiply_constant_u8_c4_in_place
);
impl_alpha_premultiply_constant_image!(
    u8,
    AC4,
    u8,
    arithmetic::alpha_premultiply_constant_u8_ac4,
    arithmetic::alpha_premultiply_constant_u8_ac4_in_place
);

#[path = "operation_impls_alpha_premultiply_constant_u16.rs"]
mod alpha_premultiply_constant_u16;

#[path = "operation_impls_alpha_comp.rs"]
mod alpha_comp;

#[path = "operation_impls_alpha_premultiply.rs"]
mod alpha_premultiply;
#[path = "operation_impls_alpha_color_key.rs"]
mod color_key;
