use crate::{
    context::StreamContext,
    error::Result,
    image::{
        morphology,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageWindow},
    },
    pipeline::image::{ImagePipeline, morphology_traits::MaskMorphologyImage},
    types::{Point, Size},
};

mod morphology_3x3_dispatch;
mod morphology_border_dispatch;

macro_rules! impl_mask_morphology_image {
    ($ty:ty, $layout:ty, $dilate:path, $erode:path) => {
        unsafe impl<'a> MaskMorphologyImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn dilate_image(
                stream_context: &StreamContext,
                source: &ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
            ) -> Result<()> {
                source.require_halo(mask_size, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $dilate(
                        stream_context,
                        &source.view(),
                        destination,
                        mask,
                        mask_size,
                        anchor,
                    )
                })
            }

            fn erode_image(
                stream_context: &StreamContext,
                source: &ImageWindow<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                mask: &[u8],
                mask_size: Size,
                anchor: Point,
            ) -> Result<()> {
                source.require_halo(mask_size, anchor)?;
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $erode(
                        stream_context,
                        &source.view(),
                        destination,
                        mask,
                        mask_size,
                        anchor,
                    )
                })
            }
        }
    };
}

impl_mask_morphology_image!(u8, C1, morphology::dilate_u8_c1, morphology::erode_u8_c1);

impl_mask_morphology_image!(u8, C3, morphology::dilate_u8_c3, morphology::erode_u8_c3);

impl_mask_morphology_image!(u8, C4, morphology::dilate_u8_c4, morphology::erode_u8_c4);

impl_mask_morphology_image!(u8, AC4, morphology::dilate_u8_ac4, morphology::erode_u8_ac4);

impl_mask_morphology_image!(u16, C1, morphology::dilate_u16_c1, morphology::erode_u16_c1);

impl_mask_morphology_image!(u16, C3, morphology::dilate_u16_c3, morphology::erode_u16_c3);

impl_mask_morphology_image!(u16, C4, morphology::dilate_u16_c4, morphology::erode_u16_c4);

impl_mask_morphology_image!(
    u16,
    AC4,
    morphology::dilate_u16_ac4,
    morphology::erode_u16_ac4
);

impl_mask_morphology_image!(f32, C1, morphology::dilate_f32_c1, morphology::erode_f32_c1);

impl_mask_morphology_image!(f32, C3, morphology::dilate_f32_c3, morphology::erode_f32_c3);

impl_mask_morphology_image!(f32, C4, morphology::dilate_f32_c4, morphology::erode_f32_c4);

impl_mask_morphology_image!(
    f32,
    AC4,
    morphology::dilate_f32_ac4,
    morphology::erode_f32_ac4
);
