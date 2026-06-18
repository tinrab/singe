use super::*;

impl_color_twist_batch!(
    color_twist_batch_u8_c1,
    C1,
    nppiColorTwistBatch32f_8u_C1R_Ctx
);
impl_color_twist_batch!(
    color_twist_batch_u8_c3,
    C3,
    nppiColorTwistBatch32f_8u_C3R_Ctx
);
impl_color_twist_batch!(
    color_twist_batch_u8_c4,
    C4,
    nppiColorTwistBatch32f_8u_C4R_Ctx
);
impl_color_twist_batch!(
    color_twist_batch_u8_ac4,
    AC4,
    nppiColorTwistBatch32f_8u_AC4R_Ctx
);
impl_color_twist_batch_in_place!(
    color_twist_batch_u8_c1_in_place,
    C1,
    nppiColorTwistBatch32f_8u_C1IR_Ctx
);
impl_color_twist_batch_in_place!(
    color_twist_batch_u8_c3_in_place,
    C3,
    nppiColorTwistBatch32f_8u_C3IR_Ctx
);
impl_color_twist_batch_in_place!(
    color_twist_batch_u8_c4_in_place,
    C4,
    nppiColorTwistBatch32f_8u_C4IR_Ctx
);
impl_color_twist_batch_in_place!(
    color_twist_batch_u8_ac4_in_place,
    AC4,
    nppiColorTwistBatch32f_8u_AC4IR_Ctx
);
impl_color_twist_batch_typed!(
    color_twist_batch_f32_c1,
    f32,
    C1,
    nppiColorTwistBatch_32f_C1R_Ctx
);
impl_color_twist_batch_typed!(
    color_twist_batch_f32_c3,
    f32,
    C3,
    nppiColorTwistBatch_32f_C3R_Ctx
);
impl_color_twist_batch_typed!(
    color_twist_batch_f32_c4,
    f32,
    C4,
    nppiColorTwistBatch_32f_C4R_Ctx
);
impl_color_twist_batch_typed!(
    color_twist_batch_f32_ac4,
    f32,
    AC4,
    nppiColorTwistBatch_32f_AC4R_Ctx
);
impl_color_twist_batch_in_place_typed!(
    color_twist_batch_f32_c1_in_place,
    f32,
    C1,
    nppiColorTwistBatch_32f_C1IR_Ctx
);
impl_color_twist_batch_in_place_typed!(
    color_twist_batch_f32_c3_in_place,
    f32,
    C3,
    nppiColorTwistBatch_32f_C3IR_Ctx
);
impl_color_twist_batch_in_place_typed!(
    color_twist_batch_f32_c4_in_place,
    f32,
    C4,
    nppiColorTwistBatch_32f_C4IR_Ctx
);
impl_color_twist_batch_in_place_typed!(
    color_twist_batch_f32_ac4_in_place,
    f32,
    AC4,
    nppiColorTwistBatch_32f_AC4IR_Ctx
);
impl_color_twist_batch_typed!(
    color_twist_batch_f16_c1,
    f16,
    C1,
    nppiColorTwistBatch32f_16f_C1R_Ctx
);
impl_color_twist_batch_typed!(
    color_twist_batch_f16_c3,
    f16,
    C3,
    nppiColorTwistBatch32f_16f_C3R_Ctx
);
impl_color_twist_batch_typed!(
    color_twist_batch_f16_c4,
    f16,
    C4,
    nppiColorTwistBatch32f_16f_C4R_Ctx
);
impl_color_twist_batch_in_place_typed!(
    color_twist_batch_f16_c1_in_place,
    f16,
    C1,
    nppiColorTwistBatch32f_16f_C1IR_Ctx
);
impl_color_twist_batch_in_place_typed!(
    color_twist_batch_f16_c3_in_place,
    f16,
    C3,
    nppiColorTwistBatch32f_16f_C3IR_Ctx
);
impl_color_twist_batch_in_place_typed!(
    color_twist_batch_f16_c4_in_place,
    f16,
    C4,
    nppiColorTwistBatch32f_16f_C4IR_Ctx
);
impl_color_twist_batch_with_constants_typed!(
    color_twist_batch_u8_c4_with_constants,
    u8,
    nppiColorTwistBatch32fC_8u_C4R_Ctx
);
impl_color_twist_batch_with_constants_in_place_typed!(
    color_twist_batch_u8_c4_with_constants_in_place,
    u8,
    nppiColorTwistBatch32fC_8u_C4IR_Ctx
);
impl_color_twist_batch_with_constants_typed!(
    color_twist_batch_f16_c4_with_constants,
    f16,
    nppiColorTwistBatch32fC_16f_C4R_Ctx
);
impl_color_twist_batch_with_constants_in_place_typed!(
    color_twist_batch_f16_c4_with_constants_in_place,
    f16,
    nppiColorTwistBatch32fC_16f_C4IR_Ctx
);
impl_color_twist_batch_with_constants_typed!(
    color_twist_batch_f32_c4_with_constants,
    f32,
    nppiColorTwistBatch_32fC_C4R_Ctx
);
impl_color_twist_batch_with_constants_in_place_typed!(
    color_twist_batch_f32_c4_with_constants_in_place,
    f32,
    nppiColorTwistBatch_32fC_C4IR_Ctx
);

pub trait ColorTwistBatch<L: ChannelLayout>: DataTypeLike {
    fn color_twist_batch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, L>],
        twists: &[ColorTwistMatrix],
        destinations: &mut [ImageViewMut<'_, Self, L>],
        min: f32,
        max: f32,
    ) -> Result<()>;
}

macro_rules! impl_color_twist_batch_dispatch {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl ColorTwistBatch<$layout> for $ty {
            fn color_twist_batch(
                stream_context: &StreamContext,
                sources: &[ImageView<'_, Self, $layout>],
                twists: &[ColorTwistMatrix],
                destinations: &mut [ImageViewMut<'_, Self, $layout>],
                min: f32,
                max: f32,
            ) -> Result<()> {
                $function(stream_context, sources, twists, destinations, min, max)
            }
        }
    };
}

impl_color_twist_batch_dispatch!(u8, C1, color_twist_batch_u8_c1);
impl_color_twist_batch_dispatch!(u8, C3, color_twist_batch_u8_c3);
impl_color_twist_batch_dispatch!(u8, C4, color_twist_batch_u8_c4);
impl_color_twist_batch_dispatch!(u8, AC4, color_twist_batch_u8_ac4);
impl_color_twist_batch_dispatch!(f16, C1, color_twist_batch_f16_c1);
impl_color_twist_batch_dispatch!(f16, C3, color_twist_batch_f16_c3);
impl_color_twist_batch_dispatch!(f16, C4, color_twist_batch_f16_c4);
impl_color_twist_batch_dispatch!(f32, C1, color_twist_batch_f32_c1);
impl_color_twist_batch_dispatch!(f32, C3, color_twist_batch_f32_c3);
impl_color_twist_batch_dispatch!(f32, C4, color_twist_batch_f32_c4);
impl_color_twist_batch_dispatch!(f32, AC4, color_twist_batch_f32_ac4);

pub fn color_twist_batch<T, L>(
    stream_context: &StreamContext,
    sources: &[ImageView<'_, T, L>],
    twists: &[ColorTwistMatrix],
    destinations: &mut [ImageViewMut<'_, T, L>],
    min: f32,
    max: f32,
) -> Result<()>
where
    T: ColorTwistBatch<L>,
    L: ChannelLayout,
{
    T::color_twist_batch(stream_context, sources, twists, destinations, min, max)
}

pub trait ColorTwistBatchInPlace<L: ChannelLayout>: DataTypeLike {
    fn color_twist_batch_in_place(
        stream_context: &StreamContext,
        images: &mut [ImageViewMut<'_, Self, L>],
        twists: &[ColorTwistMatrix],
        min: f32,
        max: f32,
    ) -> Result<()>;
}

macro_rules! impl_color_twist_batch_in_place_dispatch {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl ColorTwistBatchInPlace<$layout> for $ty {
            fn color_twist_batch_in_place(
                stream_context: &StreamContext,
                images: &mut [ImageViewMut<'_, Self, $layout>],
                twists: &[ColorTwistMatrix],
                min: f32,
                max: f32,
            ) -> Result<()> {
                $function(stream_context, images, twists, min, max)
            }
        }
    };
}

impl_color_twist_batch_in_place_dispatch!(u8, C1, color_twist_batch_u8_c1_in_place);
impl_color_twist_batch_in_place_dispatch!(u8, C3, color_twist_batch_u8_c3_in_place);
impl_color_twist_batch_in_place_dispatch!(u8, C4, color_twist_batch_u8_c4_in_place);
impl_color_twist_batch_in_place_dispatch!(u8, AC4, color_twist_batch_u8_ac4_in_place);
impl_color_twist_batch_in_place_dispatch!(f16, C1, color_twist_batch_f16_c1_in_place);
impl_color_twist_batch_in_place_dispatch!(f16, C3, color_twist_batch_f16_c3_in_place);
impl_color_twist_batch_in_place_dispatch!(f16, C4, color_twist_batch_f16_c4_in_place);
impl_color_twist_batch_in_place_dispatch!(f32, C1, color_twist_batch_f32_c1_in_place);
impl_color_twist_batch_in_place_dispatch!(f32, C3, color_twist_batch_f32_c3_in_place);
impl_color_twist_batch_in_place_dispatch!(f32, C4, color_twist_batch_f32_c4_in_place);
impl_color_twist_batch_in_place_dispatch!(f32, AC4, color_twist_batch_f32_ac4_in_place);

pub fn color_twist_batch_in_place<T, L>(
    stream_context: &StreamContext,
    images: &mut [ImageViewMut<'_, T, L>],
    twists: &[ColorTwistMatrix],
    min: f32,
    max: f32,
) -> Result<()>
where
    T: ColorTwistBatchInPlace<L>,
    L: ChannelLayout,
{
    T::color_twist_batch_in_place(stream_context, images, twists, min, max)
}

pub trait ColorTwistBatchWithConstantsC4: DataTypeLike {
    fn color_twist_batch_with_constants_c4(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C4>],
        twists: &[ColorTwistBatchConstantsMatrix4],
        destinations: &mut [ImageViewMut<'_, Self, C4>],
        min: f32,
        max: f32,
    ) -> Result<()>;
}

macro_rules! impl_color_twist_batch_with_constants_c4_dispatch {
    ($ty:ty, $function:ident) => {
        impl ColorTwistBatchWithConstantsC4 for $ty {
            fn color_twist_batch_with_constants_c4(
                stream_context: &StreamContext,
                sources: &[ImageView<'_, Self, C4>],
                twists: &[ColorTwistBatchConstantsMatrix4],
                destinations: &mut [ImageViewMut<'_, Self, C4>],
                min: f32,
                max: f32,
            ) -> Result<()> {
                $function(stream_context, sources, twists, destinations, min, max)
            }
        }
    };
}

impl_color_twist_batch_with_constants_c4_dispatch!(u8, color_twist_batch_u8_c4_with_constants);
impl_color_twist_batch_with_constants_c4_dispatch!(f16, color_twist_batch_f16_c4_with_constants);
impl_color_twist_batch_with_constants_c4_dispatch!(f32, color_twist_batch_f32_c4_with_constants);

pub fn color_twist_batch_c4_with_constants<T>(
    stream_context: &StreamContext,
    sources: &[ImageView<'_, T, C4>],
    twists: &[ColorTwistBatchConstantsMatrix4],
    destinations: &mut [ImageViewMut<'_, T, C4>],
    min: f32,
    max: f32,
) -> Result<()>
where
    T: ColorTwistBatchWithConstantsC4,
{
    T::color_twist_batch_with_constants_c4(stream_context, sources, twists, destinations, min, max)
}

pub trait ColorTwistBatchWithConstantsC4InPlace: DataTypeLike {
    fn color_twist_batch_with_constants_c4_in_place(
        stream_context: &StreamContext,
        images: &mut [ImageViewMut<'_, Self, C4>],
        twists: &[ColorTwistBatchConstantsMatrix4],
        min: f32,
        max: f32,
    ) -> Result<()>;
}

macro_rules! impl_color_twist_batch_with_constants_c4_in_place_dispatch {
    ($ty:ty, $function:ident) => {
        impl ColorTwistBatchWithConstantsC4InPlace for $ty {
            fn color_twist_batch_with_constants_c4_in_place(
                stream_context: &StreamContext,
                images: &mut [ImageViewMut<'_, Self, C4>],
                twists: &[ColorTwistBatchConstantsMatrix4],
                min: f32,
                max: f32,
            ) -> Result<()> {
                $function(stream_context, images, twists, min, max)
            }
        }
    };
}

impl_color_twist_batch_with_constants_c4_in_place_dispatch!(
    u8,
    color_twist_batch_u8_c4_with_constants_in_place
);
impl_color_twist_batch_with_constants_c4_in_place_dispatch!(
    f16,
    color_twist_batch_f16_c4_with_constants_in_place
);
impl_color_twist_batch_with_constants_c4_in_place_dispatch!(
    f32,
    color_twist_batch_f32_c4_with_constants_in_place
);

pub fn color_twist_batch_c4_with_constants_in_place<T>(
    stream_context: &StreamContext,
    images: &mut [ImageViewMut<'_, T, C4>],
    twists: &[ColorTwistBatchConstantsMatrix4],
    min: f32,
    max: f32,
) -> Result<()>
where
    T: ColorTwistBatchWithConstantsC4InPlace,
{
    T::color_twist_batch_with_constants_c4_in_place(stream_context, images, twists, min, max)
}
