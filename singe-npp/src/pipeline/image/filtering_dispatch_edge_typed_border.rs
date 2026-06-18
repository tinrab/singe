use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageView, ImageViewMut},
    },
    types::{BorderType, MaskSize, Point},
};

use super::super::ImagePipeline;
use super::super::filtering_traits::TypedEdgeDirectionalBorderFilterImage;

macro_rules! impl_typed_edge_directional_border_filter_image {
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
            TypedEdgeDirectionalBorderFilterImage<
                $source_ty,
                $source_layout,
                $destination_ty,
                $destination_layout,
            > for ImagePipeline<'a, $source_ty, $source_layout>
        {
            fn filter_sobel_horizontal_border_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $sobel_horizontal(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }

            fn filter_sobel_vertical_border_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
                mask_size: MaskSize,
                border_type: BorderType,
            ) -> Result<()> {
                $sobel_vertical(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    mask_size,
                    border_type,
                )
            }

            fn filter_scharr_horizontal_border_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
                border_type: BorderType,
            ) -> Result<()> {
                $scharr_horizontal(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }

            fn filter_scharr_vertical_border_to_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $source_ty, $source_layout>,
                source_offset: Point,
                destination: &mut ImageViewMut<'_, $destination_ty, $destination_layout>,
                border_type: BorderType,
            ) -> Result<()> {
                $scharr_vertical(
                    stream_context,
                    source,
                    source_offset,
                    destination,
                    border_type,
                )
            }
        }
    };
}

#[path = "filtering_dispatch_edge_typed_extended_border.rs"]
mod extended_border;

impl_typed_edge_directional_border_filter_image!(
    u8,
    C1,
    i16,
    C1,
    filtering::filter_sobel_horizontal_border_u8_to_i16_c1,
    filtering::filter_sobel_vertical_border_u8_to_i16_c1,
    filtering::filter_scharr_horizontal_border_u8_to_i16_c1,
    filtering::filter_scharr_vertical_border_u8_to_i16_c1
);
impl_typed_edge_directional_border_filter_image!(
    i8,
    C1,
    i16,
    C1,
    filtering::filter_sobel_horizontal_border_i8_to_i16_c1,
    filtering::filter_sobel_vertical_border_i8_to_i16_c1,
    filtering::filter_scharr_horizontal_border_i8_to_i16_c1,
    filtering::filter_scharr_vertical_border_i8_to_i16_c1
);
