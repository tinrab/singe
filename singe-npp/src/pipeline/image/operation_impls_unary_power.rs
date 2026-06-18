use super::super::operation_traits::*;
use super::ImagePipeline;

macro_rules! impl_scaled_unary_power_image {
    (
        $ty:ty,
        $layout:ty,
        $square:path,
        $square_in_place:path,
        $square_root:path,
        $square_root_in_place:path
    ) => {
        impl<'a> SquareImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn square_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $square(stream_context, source, destination, scale_factor)
            }

            fn square_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $square_in_place(stream_context, source_destination, scale_factor)
            }
        }

        impl<'a> SquareRootImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn square_root_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $square_root(stream_context, source, destination, scale_factor)
            }

            fn square_root_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $square_root_in_place(stream_context, source_destination, scale_factor)
            }
        }
    };
}

macro_rules! impl_scaled_square_image {
    ($ty:ty, $layout:ty, $square:path, $square_in_place:path) => {
        impl<'a> SquareImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn square_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $square(stream_context, source, destination, scale_factor)
            }

            fn square_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                $square_in_place(stream_context, source_destination, scale_factor)
            }
        }
    };
}

macro_rules! impl_float_unary_power_image {
    (
        $layout:ty,
        $square:path,
        $square_in_place:path,
        $square_root:path,
        $square_root_in_place:path
    ) => {
        impl<'a> SquareImage<f32, $layout> for ImagePipeline<'a, f32, $layout> {
            fn square_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $square(stream_context, source, destination)
            }

            fn square_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $square_in_place(stream_context, source_destination)
            }
        }

        impl<'a> SquareRootImage<f32, $layout> for ImagePipeline<'a, f32, $layout> {
            fn square_root_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $square_root(stream_context, source, destination)
            }

            fn square_root_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                $square_root_in_place(stream_context, source_destination)
            }
        }
    };
}

#[path = "operation_impls_ac4_unary_power.rs"]
mod ac4_impls;
#[path = "operation_impls_float_unary_power.rs"]
mod float_impls;
#[path = "operation_impls_i16_unary_power.rs"]
mod i16_impls;
#[path = "operation_impls_square_only_unary_power.rs"]
mod square_only_impls;
#[path = "operation_impls_u16_unary_power.rs"]
mod u16_impls;
#[path = "operation_impls_u8_unary_power.rs"]
mod u8_impls;
