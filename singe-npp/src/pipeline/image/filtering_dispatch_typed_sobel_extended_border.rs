use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageView, ImageViewMut},
    },
    types::{BorderType, MaskSize, Point},
};

use super::super::{ImagePipeline, filtering_traits::TypedSobelExtendedBorderFilterImage};

macro_rules! impl_typed_sobel_extended_border_filter_image {
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
            TypedSobelExtendedBorderFilterImage<
                $source_ty,
                $source_layout,
                $destination_ty,
                $destination_layout,
            > for ImagePipeline<'a, $source_ty, $source_layout>
        {
            fn filter_sobel_horizontal_second_border_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $horizontal_second(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }

            fn filter_sobel_vertical_second_border_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $vertical_second(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }

            fn filter_sobel_cross_border_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $cross(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }

            fn filter_laplace_border_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $laplace(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }
        }
    };
}

impl_typed_sobel_extended_border_filter_image!(
    u8,
    C1,
    i16,
    C1,
    filtering::filter_sobel_horizontal_second_border_u8_to_i16_c1,
    filtering::filter_sobel_vertical_second_border_u8_to_i16_c1,
    filtering::filter_sobel_cross_border_u8_to_i16_c1,
    filtering::filter_laplace_border_u8_to_i16_c1
);
impl_typed_sobel_extended_border_filter_image!(
    i8,
    C1,
    i16,
    C1,
    filtering::filter_sobel_horizontal_second_border_i8_to_i16_c1,
    filtering::filter_sobel_vertical_second_border_i8_to_i16_c1,
    filtering::filter_sobel_cross_border_i8_to_i16_c1,
    filtering::filter_laplace_border_i8_to_i16_c1
);
