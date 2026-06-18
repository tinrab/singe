use super::*;

impl_resize_sqr_pixel!(
    resize_sqr_pixel_u8_c1,
    u8,
    C1,
    nppiResizeSqrPixel_8u_C1R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_u8_c3,
    u8,
    C3,
    nppiResizeSqrPixel_8u_C3R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_u8_c4,
    u8,
    C4,
    nppiResizeSqrPixel_8u_C4R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_u8_ac4,
    u8,
    AC4,
    nppiResizeSqrPixel_8u_AC4R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_u16_c1,
    u16,
    C1,
    nppiResizeSqrPixel_16u_C1R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_u16_c3,
    u16,
    C3,
    nppiResizeSqrPixel_16u_C3R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_u16_c4,
    u16,
    C4,
    nppiResizeSqrPixel_16u_C4R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_u16_ac4,
    u16,
    AC4,
    nppiResizeSqrPixel_16u_AC4R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_i16_c1,
    i16,
    C1,
    nppiResizeSqrPixel_16s_C1R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_i16_c3,
    i16,
    C3,
    nppiResizeSqrPixel_16s_C3R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_i16_c4,
    i16,
    C4,
    nppiResizeSqrPixel_16s_C4R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_i16_ac4,
    i16,
    AC4,
    nppiResizeSqrPixel_16s_AC4R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_f32_c1,
    f32,
    C1,
    nppiResizeSqrPixel_32f_C1R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_f32_c3,
    f32,
    C3,
    nppiResizeSqrPixel_32f_C3R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_f32_c4,
    f32,
    C4,
    nppiResizeSqrPixel_32f_C4R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_f32_ac4,
    f32,
    AC4,
    nppiResizeSqrPixel_32f_AC4R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_f64_c1,
    f64,
    C1,
    nppiResizeSqrPixel_64f_C1R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_f64_c3,
    f64,
    C3,
    nppiResizeSqrPixel_64f_C3R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_f64_c4,
    f64,
    C4,
    nppiResizeSqrPixel_64f_C4R_Ctx
);
impl_resize_sqr_pixel!(
    resize_sqr_pixel_f64_ac4,
    f64,
    AC4,
    nppiResizeSqrPixel_64f_AC4R_Ctx
);

impl_generic_resize_sqr_pixel_operation!(
    ResizeSqrPixelC1,
    resize_sqr_pixel,
    resize_sqr_pixel_c1,
    C1,
    [
        u8 => resize_sqr_pixel_u8_c1,
        u16 => resize_sqr_pixel_u16_c1,
        i16 => resize_sqr_pixel_i16_c1,
        f32 => resize_sqr_pixel_f32_c1,
        f64 => resize_sqr_pixel_f64_c1,
    ]
);
impl_generic_resize_sqr_pixel_operation!(
    ResizeSqrPixelC3,
    resize_sqr_pixel,
    resize_sqr_pixel_c3,
    C3,
    [
        u8 => resize_sqr_pixel_u8_c3,
        u16 => resize_sqr_pixel_u16_c3,
        i16 => resize_sqr_pixel_i16_c3,
        f32 => resize_sqr_pixel_f32_c3,
        f64 => resize_sqr_pixel_f64_c3,
    ]
);
impl_generic_resize_sqr_pixel_operation!(
    ResizeSqrPixelC4,
    resize_sqr_pixel,
    resize_sqr_pixel_c4,
    C4,
    [
        u8 => resize_sqr_pixel_u8_c4,
        u16 => resize_sqr_pixel_u16_c4,
        i16 => resize_sqr_pixel_i16_c4,
        f32 => resize_sqr_pixel_f32_c4,
        f64 => resize_sqr_pixel_f64_c4,
    ]
);
impl_generic_resize_sqr_pixel_operation!(
    ResizeSqrPixelAc4,
    resize_sqr_pixel,
    resize_sqr_pixel_ac4,
    AC4,
    [
        u8 => resize_sqr_pixel_u8_ac4,
        u16 => resize_sqr_pixel_u16_ac4,
        i16 => resize_sqr_pixel_i16_ac4,
        f32 => resize_sqr_pixel_f32_ac4,
        f64 => resize_sqr_pixel_f64_ac4,
    ]
);

impl_resize_sqr_pixel_planar!(resize_sqr_pixel_u8_p3, u8, 3, nppiResizeSqrPixel_8u_P3R_Ctx);
impl_resize_sqr_pixel_planar!(resize_sqr_pixel_u8_p4, u8, 4, nppiResizeSqrPixel_8u_P4R_Ctx);
impl_resize_sqr_pixel_planar!(
    resize_sqr_pixel_u16_p3,
    u16,
    3,
    nppiResizeSqrPixel_16u_P3R_Ctx
);
impl_resize_sqr_pixel_planar!(
    resize_sqr_pixel_u16_p4,
    u16,
    4,
    nppiResizeSqrPixel_16u_P4R_Ctx
);
impl_resize_sqr_pixel_planar!(
    resize_sqr_pixel_i16_p3,
    i16,
    3,
    nppiResizeSqrPixel_16s_P3R_Ctx
);
impl_resize_sqr_pixel_planar!(
    resize_sqr_pixel_i16_p4,
    i16,
    4,
    nppiResizeSqrPixel_16s_P4R_Ctx
);
impl_resize_sqr_pixel_planar!(
    resize_sqr_pixel_f32_p3,
    f32,
    3,
    nppiResizeSqrPixel_32f_P3R_Ctx
);
impl_resize_sqr_pixel_planar!(
    resize_sqr_pixel_f32_p4,
    f32,
    4,
    nppiResizeSqrPixel_32f_P4R_Ctx
);
impl_resize_sqr_pixel_planar!(
    resize_sqr_pixel_f64_p3,
    f64,
    3,
    nppiResizeSqrPixel_64f_P3R_Ctx
);
impl_resize_sqr_pixel_planar!(
    resize_sqr_pixel_f64_p4,
    f64,
    4,
    nppiResizeSqrPixel_64f_P4R_Ctx
);

pub trait ResizeSqrPixelPlanar<const PLANES: usize>: DataTypeLike {
    fn resize_sqr_pixel_planar(
        stream_context: &StreamContext,
        resize: &ResizeSqrPixel,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_resize_sqr_pixel_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        impl ResizeSqrPixelPlanar<$planes> for $ty {
            fn resize_sqr_pixel_planar(
                stream_context: &StreamContext,
                resize: &ResizeSqrPixel,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut PlanarImageViewMut<'_, Self, $planes>,
            ) -> Result<()> {
                $function(stream_context, resize, source, destination)
            }
        }
    };
}

impl_resize_sqr_pixel_planar_dispatch!(u8, 3, resize_sqr_pixel_u8_p3);
impl_resize_sqr_pixel_planar_dispatch!(u8, 4, resize_sqr_pixel_u8_p4);
impl_resize_sqr_pixel_planar_dispatch!(u16, 3, resize_sqr_pixel_u16_p3);
impl_resize_sqr_pixel_planar_dispatch!(u16, 4, resize_sqr_pixel_u16_p4);
impl_resize_sqr_pixel_planar_dispatch!(i16, 3, resize_sqr_pixel_i16_p3);
impl_resize_sqr_pixel_planar_dispatch!(i16, 4, resize_sqr_pixel_i16_p4);
impl_resize_sqr_pixel_planar_dispatch!(f32, 3, resize_sqr_pixel_f32_p3);
impl_resize_sqr_pixel_planar_dispatch!(f32, 4, resize_sqr_pixel_f32_p4);
impl_resize_sqr_pixel_planar_dispatch!(f64, 3, resize_sqr_pixel_f64_p3);
impl_resize_sqr_pixel_planar_dispatch!(f64, 4, resize_sqr_pixel_f64_p4);

pub fn resize_sqr_pixel_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    resize: &ResizeSqrPixel,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: ResizeSqrPixelPlanar<PLANES>,
{
    T::resize_sqr_pixel_planar(stream_context, resize, source, destination)
}

impl_resize!(resize_u8_c1, u8, C1, nppiResize_8u_C1R_Ctx);
impl_resize!(resize_u8_c3, u8, C3, nppiResize_8u_C3R_Ctx);
impl_resize!(resize_u8_c4, u8, C4, nppiResize_8u_C4R_Ctx);
impl_resize!(resize_u8_ac4, u8, AC4, nppiResize_8u_AC4R_Ctx);
impl_resize!(resize_u16_c1, u16, C1, nppiResize_16u_C1R_Ctx);
impl_resize!(resize_u16_c3, u16, C3, nppiResize_16u_C3R_Ctx);
impl_resize!(resize_u16_c4, u16, C4, nppiResize_16u_C4R_Ctx);
impl_resize!(resize_u16_ac4, u16, AC4, nppiResize_16u_AC4R_Ctx);
impl_resize!(resize_i16_c1, i16, C1, nppiResize_16s_C1R_Ctx);
impl_resize!(resize_i16_c3, i16, C3, nppiResize_16s_C3R_Ctx);
impl_resize!(resize_i16_c4, i16, C4, nppiResize_16s_C4R_Ctx);
impl_resize!(resize_i16_ac4, i16, AC4, nppiResize_16s_AC4R_Ctx);
impl_resize!(resize_f16_c1, f16, C1, nppiResize_16f_C1R_Ctx);
impl_resize!(resize_f16_c3, f16, C3, nppiResize_16f_C3R_Ctx);
impl_resize!(resize_f16_c4, f16, C4, nppiResize_16f_C4R_Ctx);
impl_resize!(resize_f32_c1, f32, C1, nppiResize_32f_C1R_Ctx);
impl_resize!(resize_f32_c3, f32, C3, nppiResize_32f_C3R_Ctx);
impl_resize!(resize_f32_c4, f32, C4, nppiResize_32f_C4R_Ctx);
impl_resize!(resize_f32_ac4, f32, AC4, nppiResize_32f_AC4R_Ctx);

impl_generic_resize_operation!(ResizeC1, resize, resize_c1, C1, [
    u8 => resize_u8_c1,
    u16 => resize_u16_c1,
    i16 => resize_i16_c1,
    f16 => resize_f16_c1,
    f32 => resize_f32_c1,
]);
impl_generic_resize_operation!(ResizeC3, resize, resize_c3, C3, [
    u8 => resize_u8_c3,
    u16 => resize_u16_c3,
    i16 => resize_i16_c3,
    f16 => resize_f16_c3,
    f32 => resize_f32_c3,
]);
impl_generic_resize_operation!(ResizeC4, resize, resize_c4, C4, [
    u8 => resize_u8_c4,
    u16 => resize_u16_c4,
    i16 => resize_i16_c4,
    f16 => resize_f16_c4,
    f32 => resize_f32_c4,
]);
impl_generic_resize_operation!(ResizeAc4, resize, resize_ac4, AC4, [
    u8 => resize_u8_ac4,
    u16 => resize_u16_ac4,
    i16 => resize_i16_ac4,
    f32 => resize_f32_ac4,
]);

impl_resize_planar!(resize_u8_p3, u8, 3, nppiResize_8u_P3R_Ctx);
impl_resize_planar!(resize_u8_p4, u8, 4, nppiResize_8u_P4R_Ctx);
impl_resize_planar!(resize_u16_p3, u16, 3, nppiResize_16u_P3R_Ctx);
impl_resize_planar!(resize_u16_p4, u16, 4, nppiResize_16u_P4R_Ctx);
impl_resize_planar!(resize_i16_p3, i16, 3, nppiResize_16s_P3R_Ctx);
impl_resize_planar!(resize_i16_p4, i16, 4, nppiResize_16s_P4R_Ctx);
impl_resize_planar!(resize_f32_p3, f32, 3, nppiResize_32f_P3R_Ctx);
impl_resize_planar!(resize_f32_p4, f32, 4, nppiResize_32f_P4R_Ctx);

pub trait ResizePlanar<const PLANES: usize>: DataTypeLike {
    fn resize_planar(
        stream_context: &StreamContext,
        resize: &Resize,
        source: &PlanarImageView<'_, Self, PLANES>,
        destination: &mut PlanarImageViewMut<'_, Self, PLANES>,
    ) -> Result<()>;
}

macro_rules! impl_resize_planar_dispatch {
    ($ty:ty, $planes:literal, $function:ident) => {
        impl ResizePlanar<$planes> for $ty {
            fn resize_planar(
                stream_context: &StreamContext,
                resize: &Resize,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut PlanarImageViewMut<'_, Self, $planes>,
            ) -> Result<()> {
                $function(stream_context, resize, source, destination)
            }
        }
    };
}

impl_resize_planar_dispatch!(u8, 3, resize_u8_p3);
impl_resize_planar_dispatch!(u8, 4, resize_u8_p4);
impl_resize_planar_dispatch!(u16, 3, resize_u16_p3);
impl_resize_planar_dispatch!(u16, 4, resize_u16_p4);
impl_resize_planar_dispatch!(i16, 3, resize_i16_p3);
impl_resize_planar_dispatch!(i16, 4, resize_i16_p4);
impl_resize_planar_dispatch!(f32, 3, resize_f32_p3);
impl_resize_planar_dispatch!(f32, 4, resize_f32_p4);

pub fn resize_planar<T, const PLANES: usize>(
    stream_context: &StreamContext,
    resize: &Resize,
    source: &PlanarImageView<'_, T, PLANES>,
    destination: &mut PlanarImageViewMut<'_, T, PLANES>,
) -> Result<()>
where
    T: ResizePlanar<PLANES>,
{
    T::resize_planar(stream_context, resize, source, destination)
}

impl_resize_batch!(resize_batch_u8_c1, u8, C1, nppiResizeBatch_8u_C1R_Ctx);
impl_resize_batch!(resize_batch_u8_c3, u8, C3, nppiResizeBatch_8u_C3R_Ctx);
impl_resize_batch!(resize_batch_u8_c4, u8, C4, nppiResizeBatch_8u_C4R_Ctx);
impl_resize_batch!(resize_batch_u8_ac4, u8, AC4, nppiResizeBatch_8u_AC4R_Ctx);
impl_resize_batch!(resize_batch_f32_c1, f32, C1, nppiResizeBatch_32f_C1R_Ctx);
impl_resize_batch!(resize_batch_f32_c3, f32, C3, nppiResizeBatch_32f_C3R_Ctx);
impl_resize_batch!(resize_batch_f32_c4, f32, C4, nppiResizeBatch_32f_C4R_Ctx);
impl_resize_batch!(resize_batch_f32_ac4, f32, AC4, nppiResizeBatch_32f_AC4R_Ctx);

pub trait ResizeBatchOperation<L: ChannelLayout>: DataTypeLike {
    fn resize_batch(
        stream_context: &StreamContext,
        resize: &Resize,
        sources: &[ImageView<'_, Self, L>],
        destinations: &mut [ImageViewMut<'_, Self, L>],
    ) -> Result<()>;
}

macro_rules! impl_resize_batch_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl ResizeBatchOperation<$layout> for $ty {
            fn resize_batch(
                stream_context: &StreamContext,
                resize: &Resize,
                sources: &[ImageView<'_, Self, $layout>],
                destinations: &mut [ImageViewMut<'_, Self, $layout>],
            ) -> Result<()> {
                $function(stream_context, resize, sources, destinations)
            }
        }
    };
}

impl_resize_batch_operation!(u8, C1, resize_batch_u8_c1);
impl_resize_batch_operation!(u8, C3, resize_batch_u8_c3);
impl_resize_batch_operation!(u8, C4, resize_batch_u8_c4);
impl_resize_batch_operation!(u8, AC4, resize_batch_u8_ac4);
impl_resize_batch_operation!(f32, C1, resize_batch_f32_c1);
impl_resize_batch_operation!(f32, C3, resize_batch_f32_c3);
impl_resize_batch_operation!(f32, C4, resize_batch_f32_c4);
impl_resize_batch_operation!(f32, AC4, resize_batch_f32_ac4);

pub fn resize_batch<T, L>(
    stream_context: &StreamContext,
    resize: &Resize,
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageViewMut<'_, T, L>],
) -> Result<()>
where
    T: ResizeBatchOperation<L>,
    L: ChannelLayout,
{
    T::resize_batch(stream_context, resize, sources, destinations)
}

impl_resize_batch_advanced!(
    resize_batch_u8_c1_advanced,
    u8,
    C1,
    nppiResizeBatch_8u_C1R_Advanced_Ctx
);
impl_resize_batch_advanced!(
    resize_batch_u8_c3_advanced,
    u8,
    C3,
    nppiResizeBatch_8u_C3R_Advanced_Ctx
);
impl_resize_batch_advanced!(
    resize_batch_u8_c4_advanced,
    u8,
    C4,
    nppiResizeBatch_8u_C4R_Advanced_Ctx
);
impl_resize_batch_advanced!(
    resize_batch_u8_ac4_advanced,
    u8,
    AC4,
    nppiResizeBatch_8u_AC4R_Advanced_Ctx
);
impl_resize_batch_advanced!(
    resize_batch_f16_c1_advanced,
    f16,
    C1,
    nppiResizeBatch_16f_C1R_Advanced_Ctx
);
impl_resize_batch_advanced!(
    resize_batch_f16_c3_advanced,
    f16,
    C3,
    nppiResizeBatch_16f_C3R_Advanced_Ctx
);
impl_resize_batch_advanced!(
    resize_batch_f16_c4_advanced,
    f16,
    C4,
    nppiResizeBatch_16f_C4R_Advanced_Ctx
);
impl_resize_batch_advanced!(
    resize_batch_f32_c1_advanced,
    f32,
    C1,
    nppiResizeBatch_32f_C1R_Advanced_Ctx
);
impl_resize_batch_advanced!(
    resize_batch_f32_c3_advanced,
    f32,
    C3,
    nppiResizeBatch_32f_C3R_Advanced_Ctx
);
impl_resize_batch_advanced!(
    resize_batch_f32_c4_advanced,
    f32,
    C4,
    nppiResizeBatch_32f_C4R_Advanced_Ctx
);
impl_resize_batch_advanced!(
    resize_batch_f32_ac4_advanced,
    f32,
    AC4,
    nppiResizeBatch_32f_AC4R_Advanced_Ctx
);

pub trait ResizeBatchAdvancedOperation<L: ChannelLayout>: DataTypeLike {
    fn resize_batch_advanced(
        stream_context: &StreamContext,
        rois: &[ResizeBatchAdvanced],
        interpolation: InterpolationMode,
        sources: &[ImageView<'_, Self, L>],
        destinations: &mut [ImageViewMut<'_, Self, L>],
    ) -> Result<()>;
}

macro_rules! impl_resize_batch_advanced_operation {
    ($ty:ty, $layout:ty, $function:ident) => {
        impl ResizeBatchAdvancedOperation<$layout> for $ty {
            fn resize_batch_advanced(
                stream_context: &StreamContext,
                rois: &[ResizeBatchAdvanced],
                interpolation: InterpolationMode,
                sources: &[ImageView<'_, Self, $layout>],
                destinations: &mut [ImageViewMut<'_, Self, $layout>],
            ) -> Result<()> {
                $function(stream_context, rois, interpolation, sources, destinations)
            }
        }
    };
}

impl_resize_batch_advanced_operation!(u8, C1, resize_batch_u8_c1_advanced);
impl_resize_batch_advanced_operation!(u8, C3, resize_batch_u8_c3_advanced);
impl_resize_batch_advanced_operation!(u8, C4, resize_batch_u8_c4_advanced);
impl_resize_batch_advanced_operation!(u8, AC4, resize_batch_u8_ac4_advanced);
impl_resize_batch_advanced_operation!(f16, C1, resize_batch_f16_c1_advanced);
impl_resize_batch_advanced_operation!(f16, C3, resize_batch_f16_c3_advanced);
impl_resize_batch_advanced_operation!(f16, C4, resize_batch_f16_c4_advanced);
impl_resize_batch_advanced_operation!(f32, C1, resize_batch_f32_c1_advanced);
impl_resize_batch_advanced_operation!(f32, C3, resize_batch_f32_c3_advanced);
impl_resize_batch_advanced_operation!(f32, C4, resize_batch_f32_c4_advanced);
impl_resize_batch_advanced_operation!(f32, AC4, resize_batch_f32_ac4_advanced);

pub fn resize_batch_advanced<T, L>(
    stream_context: &StreamContext,
    rois: &[ResizeBatchAdvanced],
    interpolation: InterpolationMode,
    sources: &[ImageView<'_, T, L>],
    destinations: &mut [ImageViewMut<'_, T, L>],
) -> Result<()>
where
    T: ResizeBatchAdvancedOperation<L>,
    L: ChannelLayout,
{
    T::resize_batch_advanced(stream_context, rois, interpolation, sources, destinations)
}
