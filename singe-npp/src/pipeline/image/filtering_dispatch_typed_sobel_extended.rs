use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageView, ImageViewMut},
    },
    types::MaskSize,
};

use super::{ImagePipeline, filtering_traits::*};

macro_rules! impl_typed_sobel_extended_filter_image {
    (
        $source_ty:ty,
        $source_layout:ty,
        $destination_ty:ty,
        $destination_layout:ty,
        $horizontal_second:path,
        $vertical_second:path,
        $cross:path,
        $laplace:path
    ) => {
        impl<'a>
            TypedSobelExtendedFilterImage<
                $source_ty,
                $source_layout,
                $destination_ty,
                $destination_layout,
            > for ImagePipeline<'a, $source_ty, $source_layout>
        {
            fn filter_sobel_horizontal_second_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $horizontal_second(stream_context, source, destination, mask_size)
            }

            fn filter_sobel_vertical_second_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $vertical_second(stream_context, source, destination, mask_size)
            }

            fn filter_sobel_cross_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $cross(stream_context, source, destination, mask_size)
            }

            fn filter_laplace_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $laplace(stream_context, source, destination, mask_size)
            }
        }
    };
}

#[path = "filtering_dispatch_typed_sobel_extended_border.rs"]
mod border;

impl_typed_sobel_extended_filter_image!(
    u8,
    C1,
    i16,
    C1,
    filtering::filter_sobel_horizontal_second_u8_to_i16_c1,
    filtering::filter_sobel_vertical_second_u8_to_i16_c1,
    filtering::filter_sobel_cross_u8_to_i16_c1,
    filtering::filter_laplace_u8_to_i16_c1
);
impl_typed_sobel_extended_filter_image!(
    i8,
    C1,
    i16,
    C1,
    filtering::filter_sobel_horizontal_second_i8_to_i16_c1,
    filtering::filter_sobel_vertical_second_i8_to_i16_c1,
    filtering::filter_sobel_cross_i8_to_i16_c1,
    filtering::filter_laplace_i8_to_i16_c1
);
