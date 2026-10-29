use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{
        ImagePipeline,
        operation_traits::{SquareImage, SquareRootImage},
    },
};

macro_rules! impl_scaled_unary_power_image {
    (
        $ty:ty,
        $layout:ty,
        $square:path,
        $square_in_place:path,
        $square_root:path,
        $square_root_in_place:path
    ) => {
        unsafe impl<'a> SquareImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn square_image(
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
                    $square(stream_context, source, destination, scale_factor)
                })
            }

            fn square_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $square_in_place(stream_context, source_destination, scale_factor)
                })
            }
        }

        unsafe impl<'a> SquareRootImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn square_root_image(
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
                    $square_root(stream_context, source, destination, scale_factor)
                })
            }

            fn square_root_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $square_root_in_place(stream_context, source_destination, scale_factor)
                })
            }
        }
    };
}

impl_scaled_unary_power_image!(
    u8,
    AC4,
    arithmetic::square_u8_ac4,
    arithmetic::square_u8_ac4_in_place,
    arithmetic::square_root_u8_ac4,
    arithmetic::square_root_u8_ac4_in_place
);

impl_scaled_unary_power_image!(
    u16,
    AC4,
    arithmetic::square_u16_ac4,
    arithmetic::square_u16_ac4_in_place,
    arithmetic::square_root_u16_ac4,
    arithmetic::square_root_u16_ac4_in_place
);

impl_scaled_unary_power_image!(
    i16,
    AC4,
    arithmetic::square_i16_ac4,
    arithmetic::square_i16_ac4_in_place,
    arithmetic::square_root_i16_ac4,
    arithmetic::square_root_i16_ac4_in_place
);
