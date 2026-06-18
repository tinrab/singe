use super::*;

impl_mirror!(mirror_u8_c1, u8, C1, nppiMirror_8u_C1R_Ctx);
impl_mirror!(mirror_u8_c3, u8, C3, nppiMirror_8u_C3R_Ctx);
impl_mirror!(mirror_u8_c4, u8, C4, nppiMirror_8u_C4R_Ctx);
impl_mirror!(mirror_u8_ac4, u8, AC4, nppiMirror_8u_AC4R_Ctx);
impl_mirror!(mirror_u16_c1, u16, C1, nppiMirror_16u_C1R_Ctx);
impl_mirror!(mirror_u16_c3, u16, C3, nppiMirror_16u_C3R_Ctx);
impl_mirror!(mirror_u16_c4, u16, C4, nppiMirror_16u_C4R_Ctx);
impl_mirror!(mirror_u16_ac4, u16, AC4, nppiMirror_16u_AC4R_Ctx);
impl_mirror!(mirror_i16_c1, i16, C1, nppiMirror_16s_C1R_Ctx);
impl_mirror!(mirror_i16_c3, i16, C3, nppiMirror_16s_C3R_Ctx);
impl_mirror!(mirror_i16_c4, i16, C4, nppiMirror_16s_C4R_Ctx);
impl_mirror!(mirror_i16_ac4, i16, AC4, nppiMirror_16s_AC4R_Ctx);
impl_mirror!(mirror_i32_c1, i32, C1, nppiMirror_32s_C1R_Ctx);
impl_mirror!(mirror_i32_c3, i32, C3, nppiMirror_32s_C3R_Ctx);
impl_mirror!(mirror_i32_c4, i32, C4, nppiMirror_32s_C4R_Ctx);
impl_mirror!(mirror_i32_ac4, i32, AC4, nppiMirror_32s_AC4R_Ctx);
impl_mirror!(mirror_f32_c1, f32, C1, nppiMirror_32f_C1R_Ctx);
impl_mirror!(mirror_f32_c3, f32, C3, nppiMirror_32f_C3R_Ctx);
impl_mirror!(mirror_f32_c4, f32, C4, nppiMirror_32f_C4R_Ctx);
impl_mirror!(mirror_f32_ac4, f32, AC4, nppiMirror_32f_AC4R_Ctx);

impl_mirror_in_place!(mirror_u8_c1_in_place, u8, C1, nppiMirror_8u_C1IR_Ctx);
impl_mirror_in_place!(mirror_u8_c3_in_place, u8, C3, nppiMirror_8u_C3IR_Ctx);
impl_mirror_in_place!(mirror_u8_c4_in_place, u8, C4, nppiMirror_8u_C4IR_Ctx);
impl_mirror_in_place!(mirror_u8_ac4_in_place, u8, AC4, nppiMirror_8u_AC4IR_Ctx);
impl_mirror_in_place!(mirror_u16_c1_in_place, u16, C1, nppiMirror_16u_C1IR_Ctx);
impl_mirror_in_place!(mirror_u16_c3_in_place, u16, C3, nppiMirror_16u_C3IR_Ctx);
impl_mirror_in_place!(mirror_u16_c4_in_place, u16, C4, nppiMirror_16u_C4IR_Ctx);
impl_mirror_in_place!(mirror_u16_ac4_in_place, u16, AC4, nppiMirror_16u_AC4IR_Ctx);
impl_mirror_in_place!(mirror_i16_c1_in_place, i16, C1, nppiMirror_16s_C1IR_Ctx);
impl_mirror_in_place!(mirror_i16_c3_in_place, i16, C3, nppiMirror_16s_C3IR_Ctx);
impl_mirror_in_place!(mirror_i16_c4_in_place, i16, C4, nppiMirror_16s_C4IR_Ctx);
impl_mirror_in_place!(mirror_i16_ac4_in_place, i16, AC4, nppiMirror_16s_AC4IR_Ctx);
impl_mirror_in_place!(mirror_i32_c1_in_place, i32, C1, nppiMirror_32s_C1IR_Ctx);
impl_mirror_in_place!(mirror_i32_c3_in_place, i32, C3, nppiMirror_32s_C3IR_Ctx);
impl_mirror_in_place!(mirror_i32_c4_in_place, i32, C4, nppiMirror_32s_C4IR_Ctx);
impl_mirror_in_place!(mirror_i32_ac4_in_place, i32, AC4, nppiMirror_32s_AC4IR_Ctx);
impl_mirror_in_place!(mirror_f32_c1_in_place, f32, C1, nppiMirror_32f_C1IR_Ctx);
impl_mirror_in_place!(mirror_f32_c3_in_place, f32, C3, nppiMirror_32f_C3IR_Ctx);
impl_mirror_in_place!(mirror_f32_c4_in_place, f32, C4, nppiMirror_32f_C4IR_Ctx);
impl_mirror_in_place!(mirror_f32_ac4_in_place, f32, AC4, nppiMirror_32f_AC4IR_Ctx);

pub trait Mirror<L: ChannelLayout>: DataTypeLike {
    fn mirror(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, L>,
        destination: &mut ImageViewMut<'_, Self, L>,
        axis: Axis,
    ) -> Result<()>;
}

macro_rules! impl_mirror_dispatch {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl Mirror<$layout> for $ty {
            fn mirror(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $layout>,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                axis: Axis,
            ) -> Result<()> {
                $function(stream_context, source, destination, axis)
            }
        }
    };
}

impl_mirror_dispatch!(u8, C1, mirror_u8_c1);
impl_mirror_dispatch!(u8, C3, mirror_u8_c3);
impl_mirror_dispatch!(u8, C4, mirror_u8_c4);
impl_mirror_dispatch!(u8, AC4, mirror_u8_ac4);
impl_mirror_dispatch!(u16, C1, mirror_u16_c1);
impl_mirror_dispatch!(u16, C3, mirror_u16_c3);
impl_mirror_dispatch!(u16, C4, mirror_u16_c4);
impl_mirror_dispatch!(u16, AC4, mirror_u16_ac4);
impl_mirror_dispatch!(i16, C1, mirror_i16_c1);
impl_mirror_dispatch!(i16, C3, mirror_i16_c3);
impl_mirror_dispatch!(i16, C4, mirror_i16_c4);
impl_mirror_dispatch!(i16, AC4, mirror_i16_ac4);
impl_mirror_dispatch!(i32, C1, mirror_i32_c1);
impl_mirror_dispatch!(i32, C3, mirror_i32_c3);
impl_mirror_dispatch!(i32, C4, mirror_i32_c4);
impl_mirror_dispatch!(i32, AC4, mirror_i32_ac4);
impl_mirror_dispatch!(f32, C1, mirror_f32_c1);
impl_mirror_dispatch!(f32, C3, mirror_f32_c3);
impl_mirror_dispatch!(f32, C4, mirror_f32_c4);
impl_mirror_dispatch!(f32, AC4, mirror_f32_ac4);

pub fn mirror<T, L>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, L>,
    destination: &mut ImageViewMut<'_, T, L>,
    axis: Axis,
) -> Result<()>
where
    T: Mirror<L>,
    L: ChannelLayout,
{
    T::mirror(stream_context, source, destination, axis)
}

pub trait MirrorInPlace<L: ChannelLayout>: DataTypeLike {
    fn mirror_in_place(
        stream_context: &StreamContext,
        image: &mut ImageViewMut<'_, Self, L>,
        axis: Axis,
    ) -> Result<()>;
}

macro_rules! impl_mirror_in_place_dispatch {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl MirrorInPlace<$layout> for $ty {
            fn mirror_in_place(
                stream_context: &StreamContext,
                image: &mut ImageViewMut<'_, Self, $layout>,
                axis: Axis,
            ) -> Result<()> {
                $function(stream_context, image, axis)
            }
        }
    };
}

impl_mirror_in_place_dispatch!(u8, C1, mirror_u8_c1_in_place);
impl_mirror_in_place_dispatch!(u8, C3, mirror_u8_c3_in_place);
impl_mirror_in_place_dispatch!(u8, C4, mirror_u8_c4_in_place);
impl_mirror_in_place_dispatch!(u8, AC4, mirror_u8_ac4_in_place);
impl_mirror_in_place_dispatch!(u16, C1, mirror_u16_c1_in_place);
impl_mirror_in_place_dispatch!(u16, C3, mirror_u16_c3_in_place);
impl_mirror_in_place_dispatch!(u16, C4, mirror_u16_c4_in_place);
impl_mirror_in_place_dispatch!(u16, AC4, mirror_u16_ac4_in_place);
impl_mirror_in_place_dispatch!(i16, C1, mirror_i16_c1_in_place);
impl_mirror_in_place_dispatch!(i16, C3, mirror_i16_c3_in_place);
impl_mirror_in_place_dispatch!(i16, C4, mirror_i16_c4_in_place);
impl_mirror_in_place_dispatch!(i16, AC4, mirror_i16_ac4_in_place);
impl_mirror_in_place_dispatch!(i32, C1, mirror_i32_c1_in_place);
impl_mirror_in_place_dispatch!(i32, C3, mirror_i32_c3_in_place);
impl_mirror_in_place_dispatch!(i32, C4, mirror_i32_c4_in_place);
impl_mirror_in_place_dispatch!(i32, AC4, mirror_i32_ac4_in_place);
impl_mirror_in_place_dispatch!(f32, C1, mirror_f32_c1_in_place);
impl_mirror_in_place_dispatch!(f32, C3, mirror_f32_c3_in_place);
impl_mirror_in_place_dispatch!(f32, C4, mirror_f32_c4_in_place);
impl_mirror_in_place_dispatch!(f32, AC4, mirror_f32_ac4_in_place);

pub fn mirror_in_place<T, L>(
    stream_context: &StreamContext,
    image: &mut ImageViewMut<'_, T, L>,
    axis: Axis,
) -> Result<()>
where
    T: MirrorInPlace<L>,
    L: ChannelLayout,
{
    T::mirror_in_place(stream_context, image, axis)
}

impl_mirror_batch!(mirror_batch_f32_c1, f32, C1, nppiMirrorBatch_32f_C1R_Ctx);
impl_mirror_batch!(mirror_batch_f32_c3, f32, C3, nppiMirrorBatch_32f_C3R_Ctx);
impl_mirror_batch!(mirror_batch_f32_c4, f32, C4, nppiMirrorBatch_32f_C4R_Ctx);
impl_mirror_batch!(mirror_batch_f32_ac4, f32, AC4, nppiMirrorBatch_32f_AC4R_Ctx);
impl_mirror_batch_in_place!(
    mirror_batch_f32_c1_in_place,
    f32,
    C1,
    nppiMirrorBatch_32f_C1IR_Ctx
);
impl_mirror_batch_in_place!(
    mirror_batch_f32_c3_in_place,
    f32,
    C3,
    nppiMirrorBatch_32f_C3IR_Ctx
);
impl_mirror_batch_in_place!(
    mirror_batch_f32_c4_in_place,
    f32,
    C4,
    nppiMirrorBatch_32f_C4IR_Ctx
);
impl_mirror_batch_in_place!(
    mirror_batch_f32_ac4_in_place,
    f32,
    AC4,
    nppiMirrorBatch_32f_AC4IR_Ctx
);

pub trait MirrorBatch<L: ChannelLayout>: DataTypeLike {
    fn mirror_batch(
        stream_context: &StreamContext,
        axis: Axis,
        sources: &[ImageView<'_, Self, L>],
        destinations: &mut [ImageViewMut<'_, Self, L>],
    ) -> Result<()>;
}

macro_rules! impl_mirror_batch_dispatch {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl MirrorBatch<$layout> for $ty {
            fn mirror_batch(
                stream_context: &StreamContext,
                axis: Axis,
                sources: &[ImageView<'_, Self, $layout>],
                destinations: &mut [ImageViewMut<'_, Self, $layout>],
            ) -> Result<()> {
                $function(stream_context, axis, sources, destinations)
            }
        }
    };
}

impl_mirror_batch_dispatch!(f32, C1, mirror_batch_f32_c1);
impl_mirror_batch_dispatch!(f32, C3, mirror_batch_f32_c3);
impl_mirror_batch_dispatch!(f32, C4, mirror_batch_f32_c4);
impl_mirror_batch_dispatch!(f32, AC4, mirror_batch_f32_ac4);

pub fn mirror_batch<T, L>(
    stream_context: &StreamContext,
    axis: Axis,
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageViewMut<'_, T, L>],
) -> Result<()>
where
    T: MirrorBatch<L>,
    L: ChannelLayout,
{
    T::mirror_batch(stream_context, axis, sources, destinations)
}

pub trait MirrorBatchInPlace<L: ChannelLayout>: DataTypeLike {
    fn mirror_batch_in_place(
        stream_context: &StreamContext,
        axis: Axis,
        images: &mut [ImageViewMut<'_, Self, L>],
    ) -> Result<()>;
}

macro_rules! impl_mirror_batch_in_place_dispatch {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl MirrorBatchInPlace<$layout> for $ty {
            fn mirror_batch_in_place(
                stream_context: &StreamContext,
                axis: Axis,
                images: &mut [ImageViewMut<'_, Self, $layout>],
            ) -> Result<()> {
                $function(stream_context, axis, images)
            }
        }
    };
}

impl_mirror_batch_in_place_dispatch!(f32, C1, mirror_batch_f32_c1_in_place);
impl_mirror_batch_in_place_dispatch!(f32, C3, mirror_batch_f32_c3_in_place);
impl_mirror_batch_in_place_dispatch!(f32, C4, mirror_batch_f32_c4_in_place);
impl_mirror_batch_in_place_dispatch!(f32, AC4, mirror_batch_f32_ac4_in_place);

pub fn mirror_batch_in_place<T, L>(
    stream_context: &StreamContext,
    axis: Axis,
    images: &mut [ImageViewMut<'_, T, L>],
) -> Result<()>
where
    T: MirrorBatchInPlace<L>,
    L: ChannelLayout,
{
    T::mirror_batch_in_place(stream_context, axis, images)
}
