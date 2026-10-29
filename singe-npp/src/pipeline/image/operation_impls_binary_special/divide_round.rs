use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{AC4, C1, C3, C4, ImageOutputView, ImageView, ImageViewMut},
    },
    pipeline::image::{ImagePipeline, operation_traits::DivideRoundImage},
    types::RoundMode,
};

macro_rules! impl_divide_round_image {
    ($ty:ty, $layout:ty, $divide:path, $divide_in_place:path) => {
        unsafe impl<'a> DivideRoundImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn divide_round_image(
                stream_context: &StreamContext,
                left: &ImageView<'_, $ty, $layout>,
                right: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageOutputView<'_, $ty, $layout>,
                scale_factor: i32,
                round_mode: RoundMode,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(left)?;
                stream_context.prepare_region(right)?;
                stream_context.prepare_region(destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $divide(
                        stream_context,
                        left,
                        right,
                        destination,
                        scale_factor,
                        round_mode,
                    )
                })
            }

            fn divide_round_image_in_place(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
                scale_factor: i32,
                round_mode: RoundMode,
            ) -> Result<()> {
                stream_context.stream().ensure_not_capturing()?;
                stream_context.prepare_region(source)?;
                stream_context.prepare_region(source_destination)?;
                // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
                stream_context.sync_scope(|| unsafe {
                    $divide_in_place(
                        stream_context,
                        source,
                        source_destination,
                        scale_factor,
                        round_mode,
                    )
                })
            }
        }
    };
}

impl_divide_round_image!(
    u8,
    C1,
    arithmetic::divide_round_u8_c1,
    arithmetic::divide_round_u8_c1_in_place
);

impl_divide_round_image!(
    u8,
    C3,
    arithmetic::divide_round_u8_c3,
    arithmetic::divide_round_u8_c3_in_place
);

impl_divide_round_image!(
    u8,
    C4,
    arithmetic::divide_round_u8_c4,
    arithmetic::divide_round_u8_c4_in_place
);

impl_divide_round_image!(
    u8,
    AC4,
    arithmetic::divide_round_u8_ac4,
    arithmetic::divide_round_u8_ac4_in_place
);

impl_divide_round_image!(
    u16,
    C1,
    arithmetic::divide_round_u16_c1,
    arithmetic::divide_round_u16_c1_in_place
);

impl_divide_round_image!(
    u16,
    C3,
    arithmetic::divide_round_u16_c3,
    arithmetic::divide_round_u16_c3_in_place
);

impl_divide_round_image!(
    u16,
    C4,
    arithmetic::divide_round_u16_c4,
    arithmetic::divide_round_u16_c4_in_place
);

impl_divide_round_image!(
    u16,
    AC4,
    arithmetic::divide_round_u16_ac4,
    arithmetic::divide_round_u16_ac4_in_place
);

impl_divide_round_image!(
    i16,
    C1,
    arithmetic::divide_round_i16_c1,
    arithmetic::divide_round_i16_c1_in_place
);

impl_divide_round_image!(
    i16,
    C3,
    arithmetic::divide_round_i16_c3,
    arithmetic::divide_round_i16_c3_in_place
);

impl_divide_round_image!(
    i16,
    C4,
    arithmetic::divide_round_i16_c4,
    arithmetic::divide_round_i16_c4_in_place
);

impl_divide_round_image!(
    i16,
    AC4,
    arithmetic::divide_round_i16_ac4,
    arithmetic::divide_round_i16_ac4_in_place
);
