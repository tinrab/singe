use crate::{
    context::StreamContext,
    error::Result,
    image::{
        morphology,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, morphology_traits::Morphology3x3Image};

macro_rules! impl_morphology_3x3_image {
    ($ty:ty, $layout:ty, $dilate:path, $erode:path) => {
        impl<'a> Morphology3x3Image<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn dilate_3x3_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $dilate(stream_context, source, destination)
            }

            fn erode_3x3_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $erode(stream_context, source, destination)
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
