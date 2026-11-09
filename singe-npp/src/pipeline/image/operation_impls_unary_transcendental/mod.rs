use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{ImagePipeline, operation_traits::UnaryTranscendentalImage},
};

macro_rules! impl_scaled_unary_transcendental_image {
    (
        $ty:ty,
        $layout:ty,
        $natural_logarithm:path,
        $natural_logarithm_in_place:path,
        $exponential:path,
        $exponential_in_place:path
    ) => {
        unsafe impl<'a> UnaryTranscendentalImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn natural_logarithm_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $natural_logarithm(stream_context, source, destination, scale_factor)
                })
            }

            fn natural_logarithm_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $natural_logarithm_in_place(stream_context, source_destination, scale_factor)
                })
            }

            fn exponential_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $exponential(stream_context, source, destination, scale_factor)
                })
            }

            fn exponential_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $exponential_in_place(stream_context, source_destination, scale_factor)
                })
            }
        }
    };
}

macro_rules! impl_float_unary_transcendental_image {
    (
        $layout:ty,
        $natural_logarithm:path,
        $natural_logarithm_in_place:path,
        $exponential:path,
        $exponential_in_place:path
    ) => {
        unsafe impl<'a> UnaryTranscendentalImage<f32, $layout> for ImagePipeline<'a, f32, $layout> {
            fn natural_logarithm_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $natural_logarithm(stream_context, source, destination)
                })
            }

            fn natural_logarithm_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $natural_logarithm_in_place(stream_context, source_destination)
                })
            }

            fn exponential_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, f32, $layout>,
                destination: &mut ImageOutputView<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $exponential(stream_context, source, destination) })
            }

            fn exponential_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, f32, $layout>,
                _scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $exponential_in_place(stream_context, source_destination)
                })
            }
        }
    };
}

impl_scaled_unary_transcendental_image!(
    u8,
    C1,
    arithmetic::natural_logarithm_u8_c1,
    arithmetic::natural_logarithm_u8_c1_in_place,
    arithmetic::exponential_u8_c1,
    arithmetic::exponential_u8_c1_in_place
);

impl_scaled_unary_transcendental_image!(
    u8,
    C3,
    arithmetic::natural_logarithm_u8_c3,
    arithmetic::natural_logarithm_u8_c3_in_place,
    arithmetic::exponential_u8_c3,
    arithmetic::exponential_u8_c3_in_place
);

mod f16_natural_logarithm;
mod f32_unary_transcendental;
mod i16_unary_transcendental;
mod u16_unary_transcendental;
