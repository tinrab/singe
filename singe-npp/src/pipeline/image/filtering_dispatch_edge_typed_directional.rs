use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageView, ImageViewMut},
    },
    types::MaskSize,
};

use super::super::{ImagePipeline, filtering_traits::TypedEdgeDirectionalFilterImage};

macro_rules! impl_typed_edge_directional_filter_image {
    (
        $source_ty:ty,
        $source_layout:ty,
        $destination_ty:ty,
        $destination_layout:ty,
        $sobel_horizontal:path,
        $sobel_vertical:path,
        $scharr_horizontal:path,
        $scharr_vertical:path
    ) => {
        impl<'a>
            TypedEdgeDirectionalFilterImage<
                $source_ty,
                $source_layout,
                $destination_ty,
                $destination_layout,
            > for ImagePipeline<'a, $source_ty, $source_layout>
        {
            fn filter_sobel_horizontal_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $sobel_horizontal(stream_context, source, destination, mask_size)
            }

            fn filter_sobel_vertical_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $sobel_vertical(stream_context, source, destination, mask_size)
            }

            fn filter_scharr_horizontal_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
            ) -> Result<()> {
                $scharr_horizontal(stream_context, source, destination)
            }

            fn filter_scharr_vertical_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
            ) -> Result<()> {
                $scharr_vertical(stream_context, source, destination)
            }
        }
    };
}

impl_typed_edge_directional_filter_image!(
    u8,
    C1,
    i16,
    C1,
    filtering::filter_sobel_horizontal_u8_to_i16_c1,
    filtering::filter_sobel_vertical_u8_to_i16_c1,
    filtering::filter_scharr_horizontal_u8_to_i16_c1,
    filtering::filter_scharr_vertical_u8_to_i16_c1
);
impl_typed_edge_directional_filter_image!(
    i8,
    C1,
    i16,
    C1,
    filtering::filter_sobel_horizontal_i8_to_i16_c1,
    filtering::filter_sobel_vertical_i8_to_i16_c1,
    filtering::filter_scharr_horizontal_i8_to_i16_c1,
    filtering::filter_scharr_vertical_i8_to_i16_c1
);
