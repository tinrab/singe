use crate::{
    context::StreamContext,
    error::Result,
    image::{
        morphology,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageWindow},
    },
    pipeline::image::{ImagePipeline, morphology_traits::Morphology3x3Image},
    types::{Point, Size},
};

macro_rules! impl_morphology_3x3_image {
    ($ty:ty, $layout:ty, $dilate:path, $erode:path) => {
        unsafe impl<'a> Morphology3x3Image<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn dilate_3x3_image(
                stream_context: &StreamContext,
                source: &ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                source.require_halo(Size::new(3, 3), Point { x: 1, y: 1 })?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $dilate(stream_context, &source.view(), destination) })
            }

            fn erode_3x3_image(
                stream_context: &StreamContext,
                source: &ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
            ) -> Result<()> {
                source.require_halo(Size::new(3, 3), Point { x: 1, y: 1 })?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $erode(stream_context, &source.view(), destination) })
            }
        }
    };
}

impl_morphology_3x3_image!(
    u8,
    C1,
    morphology::dilate_3x3_u8_c1,
    morphology::erode_3x3_u8_c1
);

impl_morphology_3x3_image!(
    u8,
    C3,
    morphology::dilate_3x3_u8_c3,
    morphology::erode_3x3_u8_c3
);

impl_morphology_3x3_image!(
    u8,
    C4,
    morphology::dilate_3x3_u8_c4,
    morphology::erode_3x3_u8_c4
);

impl_morphology_3x3_image!(
    u8,
    AC4,
    morphology::dilate_3x3_u8_ac4,
    morphology::erode_3x3_u8_ac4
);

impl_morphology_3x3_image!(
    u16,
    C1,
    morphology::dilate_3x3_u16_c1,
    morphology::erode_3x3_u16_c1
);

impl_morphology_3x3_image!(
    u16,
    C3,
    morphology::dilate_3x3_u16_c3,
    morphology::erode_3x3_u16_c3
);

impl_morphology_3x3_image!(
    u16,
    C4,
    morphology::dilate_3x3_u16_c4,
    morphology::erode_3x3_u16_c4
);

impl_morphology_3x3_image!(
    u16,
    AC4,
    morphology::dilate_3x3_u16_ac4,
    morphology::erode_3x3_u16_ac4
);

impl_morphology_3x3_image!(
    f32,
    C1,
    morphology::dilate_3x3_f32_c1,
    morphology::erode_3x3_f32_c1
);

impl_morphology_3x3_image!(
    f32,
    C3,
    morphology::dilate_3x3_f32_c3,
    morphology::erode_3x3_f32_c3
);

impl_morphology_3x3_image!(
    f32,
    C4,
    morphology::dilate_3x3_f32_c4,
    morphology::erode_3x3_f32_c4
);

impl_morphology_3x3_image!(
    f32,
    AC4,
    morphology::dilate_3x3_f32_ac4,
    morphology::erode_3x3_f32_ac4
);

impl_morphology_3x3_image!(
    f64,
    C1,
    morphology::dilate_3x3_f64_c1,
    morphology::erode_3x3_f64_c1
);
