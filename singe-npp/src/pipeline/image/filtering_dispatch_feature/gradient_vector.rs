use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, C3, ImageOutputView, ImageView},
    },
    pipeline::image::{ImagePipeline, filtering_traits::GradientVectorBorderImage},
    types::{BorderType, ImageNormalization, MaskSize, Point},
};

macro_rules! impl_gradient_vector_border_image {
    ($ty:ty, $layout:ty, $gradient_ty:ty, $prewitt:path, $scharr:path, $sobel:path) => {
        unsafe impl<'a> GradientVectorBorderImage<$ty, $layout, $gradient_ty>
            for ImagePipeline<'a, $ty, $layout>
        {
            fn gradient_vector_prewitt_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination_x: &mut ImageOutputView<'_, $gradient_ty, C1>,
                destination_y: &mut ImageOutputView<'_, $gradient_ty, C1>,
                destination_magnitude: &mut ImageOutputView<'_, $gradient_ty, C1>,
                destination_angle: &mut ImageOutputView<'_, f32, C1>,
                mask_size: MaskSize,
                norm: ImageNormalization,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination_x)?;
                stream_context.prepare_region(destination_y)?;
                stream_context.prepare_region(destination_magnitude)?;
                stream_context.prepare_region(destination_angle)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $prewitt(
                        stream_context,
                        source,
                        source_offset,
                        Some(destination_x),
                        Some(destination_y),
                        Some(destination_magnitude),
                        Some(destination_angle),
                        mask_size,
                        norm,
                        border_type,
                    )
                })
            }

            fn gradient_vector_scharr_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination_x: &mut ImageOutputView<'_, $gradient_ty, C1>,
                destination_y: &mut ImageOutputView<'_, $gradient_ty, C1>,
                destination_magnitude: &mut ImageOutputView<'_, $gradient_ty, C1>,
                destination_angle: &mut ImageOutputView<'_, f32, C1>,
                mask_size: MaskSize,
                norm: ImageNormalization,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination_x)?;
                stream_context.prepare_region(destination_y)?;
                stream_context.prepare_region(destination_magnitude)?;
                stream_context.prepare_region(destination_angle)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $scharr(
                        stream_context,
                        source,
                        source_offset,
                        Some(destination_x),
                        Some(destination_y),
                        Some(destination_magnitude),
                        Some(destination_angle),
                        mask_size,
                        norm,
                        border_type,
                    )
                })
            }

            fn gradient_vector_sobel_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                destination_x: &mut ImageOutputView<'_, $gradient_ty, C1>,
                destination_y: &mut ImageOutputView<'_, $gradient_ty, C1>,
                destination_magnitude: &mut ImageOutputView<'_, $gradient_ty, C1>,
                destination_angle: &mut ImageOutputView<'_, f32, C1>,
                mask_size: MaskSize,
                norm: ImageNormalization,
                border_type: BorderType,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination_x)?;
                stream_context.prepare_region(destination_y)?;
                stream_context.prepare_region(destination_magnitude)?;
                stream_context.prepare_region(destination_angle)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $sobel(
                        stream_context,
                        source,
                        source_offset,
                        Some(destination_x),
                        Some(destination_y),
                        Some(destination_magnitude),
                        Some(destination_angle),
                        mask_size,
                        norm,
                        border_type,
                    )
                })
            }
        }
    };
}

mod f32;
mod u8_i16;

impl_gradient_vector_border_image!(
    i16,
    C1,
    f32,
    filtering::gradient_vector_prewitt_border_i16_to_f32_c1,
    filtering::gradient_vector_scharr_border_i16_to_f32_c1,
    filtering::gradient_vector_sobel_border_i16_to_f32_c1
);

impl_gradient_vector_border_image!(
    i16,
    C3,
    f32,
    filtering::gradient_vector_prewitt_border_i16_c3_to_f32_c1,
    filtering::gradient_vector_scharr_border_i16_c3_to_f32_c1,
    filtering::gradient_vector_sobel_border_i16_c3_to_f32_c1
);

impl_gradient_vector_border_image!(
    u16,
    C1,
    f32,
    filtering::gradient_vector_prewitt_border_u16_to_f32_c1,
    filtering::gradient_vector_scharr_border_u16_to_f32_c1,
    filtering::gradient_vector_sobel_border_u16_to_f32_c1
);

impl_gradient_vector_border_image!(
    u16,
    C3,
    f32,
    filtering::gradient_vector_prewitt_border_u16_c3_to_f32_c1,
    filtering::gradient_vector_scharr_border_u16_c3_to_f32_c1,
    filtering::gradient_vector_sobel_border_u16_c3_to_f32_c1
);
