use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageOutputView, ImageWindow},
    },
    pipeline::image::{ImagePipeline, filtering_traits::TypedEdgeDirectionalFilterImage},
    types::MaskSize,
};

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
        unsafe impl<'a>
            TypedEdgeDirectionalFilterImage<
                $source_ty,
                $source_layout,
                $destination_ty,
                $destination_layout,
            > for ImagePipeline<'a, $source_ty, $source_layout>
        {
            fn filter_sobel_horizontal_to_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $source_ty, $source_layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $destination_layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                let (mask, anchor) =
                    $crate::pipeline::image::filtering::centered_filter_geometry(mask_size)?;
                source.require_halo(mask, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $sobel_horizontal(stream_context, &source.view(), destination, mask_size)
                })
            }

            fn filter_sobel_vertical_to_image(
                stream_context: &StreamContext,
                source: &$crate::image::view::ImageWindow<'_, $source_ty, $source_layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $destination_layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                let (mask, anchor) =
                    $crate::pipeline::image::filtering::centered_filter_geometry(mask_size)?;
                source.require_halo(mask, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $sobel_vertical(stream_context, &source.view(), destination, mask_size)
                })
            }

            fn filter_scharr_horizontal_to_image(
                stream_context: &StreamContext,
                source: &ImageWindow<'_, $source_ty, $source_layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $destination_layout>,
            ) -> Result<()> {
                source.require_halo(
                    $crate::types::Size::new(3, 3),
                    $crate::types::Point { x: 1, y: 1 },
                )?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $scharr_horizontal(stream_context, &source.view(), destination)
                })
            }

            fn filter_scharr_vertical_to_image(
                stream_context: &StreamContext,
                source: &ImageWindow<'_, $source_ty, $source_layout>,
                destination: &mut ImageOutputView<'_, $destination_ty, $destination_layout>,
            ) -> Result<()> {
                source.require_halo(
                    $crate::types::Size::new(3, 3),
                    $crate::types::Point { x: 1, y: 1 },
                )?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $scharr_vertical(stream_context, &source.view(), destination)
                })
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
