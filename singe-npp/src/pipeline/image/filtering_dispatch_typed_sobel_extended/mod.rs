use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageOutputView},
    },
    pipeline::image::{ImagePipeline, filtering_traits::TypedSobelExtendedFilterImage},
    types::MaskSize,
};

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
        unsafe impl<'a>
            TypedSobelExtendedFilterImage<
                $source_ty,
                $source_layout,
                $destination_ty,
                $destination_layout,
            > for ImagePipeline<'a, $source_ty, $source_layout>
        {
            fn filter_sobel_horizontal_second_to_image(
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
                    $horizontal_second(stream_context, &source.view(), destination, mask_size)
                })
            }

            fn filter_sobel_vertical_second_to_image(
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
                    $vertical_second(stream_context, &source.view(), destination, mask_size)
                })
            }

            fn filter_sobel_cross_to_image(
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
                    $cross(stream_context, &source.view(), destination, mask_size)
                })
            }

            fn filter_laplace_to_image(
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
                    $laplace(stream_context, &source.view(), destination, mask_size)
                })
            }
        }
    };
}

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
