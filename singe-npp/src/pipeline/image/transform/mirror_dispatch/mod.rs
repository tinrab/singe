use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::transform::{ImagePipeline, MirrorImage},
    types::Axis,
};

macro_rules! impl_mirror_image {
    ($ty:ty, $layout:ty, $mirror:path, $mirror_in_place:path) => {
        unsafe impl<'a> MirrorImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn mirror_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                axis: Axis,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(destination)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $mirror(stream_context, source, destination, axis) })
            }

            fn mirror_image_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, $ty, $layout>,
                axis: Axis,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(image)?;
                stream_context
                    // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                    .sync_scope(|| unsafe { $mirror_in_place(stream_context, image, axis) })
            }
        }
    };
}

mod f32_dispatch;

impl_mirror_image!(
    u8,
    C1,
    geometry::mirror_u8_c1,
    geometry::mirror_u8_c1_in_place
);

impl_mirror_image!(
    u8,
    C3,
    geometry::mirror_u8_c3,
    geometry::mirror_u8_c3_in_place
);

impl_mirror_image!(
    u8,
    C4,
    geometry::mirror_u8_c4,
    geometry::mirror_u8_c4_in_place
);

impl_mirror_image!(
    u8,
    AC4,
    geometry::mirror_u8_ac4,
    geometry::mirror_u8_ac4_in_place
);

impl_mirror_image!(
    u16,
    C1,
    geometry::mirror_u16_c1,
    geometry::mirror_u16_c1_in_place
);

impl_mirror_image!(
    u16,
    C3,
    geometry::mirror_u16_c3,
    geometry::mirror_u16_c3_in_place
);

impl_mirror_image!(
    u16,
    C4,
    geometry::mirror_u16_c4,
    geometry::mirror_u16_c4_in_place
);

impl_mirror_image!(
    u16,
    AC4,
    geometry::mirror_u16_ac4,
    geometry::mirror_u16_ac4_in_place
);

impl_mirror_image!(
    i16,
    C1,
    geometry::mirror_i16_c1,
    geometry::mirror_i16_c1_in_place
);

impl_mirror_image!(
    i16,
    C3,
    geometry::mirror_i16_c3,
    geometry::mirror_i16_c3_in_place
);

impl_mirror_image!(
    i16,
    C4,
    geometry::mirror_i16_c4,
    geometry::mirror_i16_c4_in_place
);

impl_mirror_image!(
    i16,
    AC4,
    geometry::mirror_i16_ac4,
    geometry::mirror_i16_ac4_in_place
);

impl_mirror_image!(
    i32,
    C1,
    geometry::mirror_i32_c1,
    geometry::mirror_i32_c1_in_place
);

impl_mirror_image!(
    i32,
    C3,
    geometry::mirror_i32_c3,
    geometry::mirror_i32_c3_in_place
);

impl_mirror_image!(
    i32,
    C4,
    geometry::mirror_i32_c4,
    geometry::mirror_i32_c4_in_place
);

impl_mirror_image!(
    i32,
    AC4,
    geometry::mirror_i32_ac4,
    geometry::mirror_i32_ac4_in_place
);
