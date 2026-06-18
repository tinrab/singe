use super::*;

impl_color_convert_planar_to_planar!(rgb_to_yuv_u8_p3, u8, 3, nppiRGBToYUV_8u_P3R_Ctx);
impl_color_convert_same_layout!(bgr_to_yuv_u8_c3, u8, C3, nppiBGRToYUV_8u_C3R_Ctx);
impl_color_convert_same_layout!(bgr_to_yuv_u8_ac4, u8, AC4, nppiBGRToYUV_8u_AC4R_Ctx);
impl_generic_color_convert_same_layout!(BgrToYuvC3, bgr_to_yuv, bgr_to_yuv_c3, C3, [
    u8 => bgr_to_yuv_u8_c3,
]);
impl_generic_color_convert_same_layout!(BgrToYuvAC4, bgr_to_yuv, bgr_to_yuv_ac4, AC4, [
    u8 => bgr_to_yuv_u8_ac4,
]);
impl_color_convert_planar_to_planar!(bgr_to_yuv_u8_p3, u8, 3, nppiBGRToYUV_8u_P3R_Ctx);
impl_color_convert_planar_to_planar!(yuv_to_rgb_u8_p3, u8, 3, nppiYUVToRGB_8u_P3R_Ctx);
impl_color_convert_same_layout!(yuv_to_bgr_u8_c3, u8, C3, nppiYUVToBGR_8u_C3R_Ctx);
impl_color_convert_same_layout!(yuv_to_bgr_u8_ac4, u8, AC4, nppiYUVToBGR_8u_AC4R_Ctx);
impl_generic_color_convert_same_layout!(YuvToBgrC3, yuv_to_bgr, yuv_to_bgr_c3, C3, [
    u8 => yuv_to_bgr_u8_c3,
]);
impl_generic_color_convert_same_layout!(YuvToBgrAC4, yuv_to_bgr, yuv_to_bgr_ac4, AC4, [
    u8 => yuv_to_bgr_u8_ac4,
]);
impl_color_convert_batch_same_layout!(yuv_to_bgr_batch_u8_c3, u8, C3, nppiYUVToBGRBatch_8u_C3R_Ctx);
impl_color_convert_batch_same_layout_advanced!(
    yuv_to_bgr_batch_u8_c3_advanced,
    u8,
    C3,
    nppiYUVToBGRBatch_8u_C3R_Advanced_Ctx
);

macro_rules! impl_color_convert_batch_c3_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                sources: &[ImageView<'_, Self, C3>],
                destinations: &mut [ImageViewMut<'_, Self, C3>],
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, T, C3>],
            destinations: &mut [ImageViewMut<'_, T, C3>],
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, sources, destinations)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    sources: &[ImageView<'_, Self, C3>],
                    destinations: &mut [ImageViewMut<'_, Self, C3>],
                ) -> Result<()> {
                    $direct(stream_context, sources, destinations)
                }
            }
        )*
    };
}

impl_color_convert_batch_c3_dispatch!(YuvToRgbBatchC3, yuv_to_rgb_batch, yuv_to_rgb_batch_c3, [
    u8 => yuv_to_rgb_batch_u8_c3,
]);
impl_color_convert_batch_c3_dispatch!(
    YuvToRgbBatchAdvancedC3,
    yuv_to_rgb_batch_advanced,
    yuv_to_rgb_batch_c3_advanced,
    [u8 => yuv_to_rgb_batch_u8_c3_advanced]
);
impl_color_convert_batch_c3_dispatch!(
    YcbcrToRgbBatchC3,
    ycbcr_to_rgb_batch,
    ycbcr_to_rgb_batch_c3,
    [u8 => ycbcr_to_rgb_batch_u8_c3]
);
impl_color_convert_batch_c3_dispatch!(
    YcbcrToRgbBatchAdvancedC3,
    ycbcr_to_rgb_batch_advanced,
    ycbcr_to_rgb_batch_c3_advanced,
    [u8 => ycbcr_to_rgb_batch_u8_c3_advanced]
);
impl_color_convert_batch_c3_dispatch!(
    YcbcrToBgrBatchC3,
    ycbcr_to_bgr_batch,
    ycbcr_to_bgr_batch_c3,
    [u8 => ycbcr_to_bgr_batch_u8_c3]
);
impl_color_convert_batch_c3_dispatch!(
    YcbcrToBgrBatchAdvancedC3,
    ycbcr_to_bgr_batch_advanced,
    ycbcr_to_bgr_batch_c3_advanced,
    [u8 => ycbcr_to_bgr_batch_u8_c3_advanced]
);
impl_color_convert_batch_c3_dispatch!(YuvToBgrBatchC3, yuv_to_bgr_batch, yuv_to_bgr_batch_c3, [
    u8 => yuv_to_bgr_batch_u8_c3,
]);
impl_color_convert_batch_c3_dispatch!(
    YuvToBgrBatchAdvancedC3,
    yuv_to_bgr_batch_advanced,
    yuv_to_bgr_batch_c3_advanced,
    [u8 => yuv_to_bgr_batch_u8_c3_advanced]
);

impl_color_convert_batch_planar_to_packed!(
    yuv_to_bgr_batch_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiYUVToBGRBatch_8u_P3C3R_Ctx
);
impl_color_convert_batch_planar_to_packed_advanced!(
    yuv_to_bgr_batch_u8_p3_to_c3_advanced,
    u8,
    3,
    C3,
    nppiYUVToBGRBatch_8u_P3C3R_Advanced_Ctx
);

macro_rules! impl_color_convert_batch_p3_to_c3_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                sources: &[PlanarImageView<'_, Self, 3>],
                destinations: &mut [ImageViewMut<'_, Self, C3>],
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            sources: &[PlanarImageView<'_, T, 3>],
            destinations: &mut [ImageViewMut<'_, T, C3>],
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, sources, destinations)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    sources: &[PlanarImageView<'_, Self, 3>],
                    destinations: &mut [ImageViewMut<'_, Self, C3>],
                ) -> Result<()> {
                    $direct(stream_context, sources, destinations)
                }
            }
        )*
    };
}

impl_color_convert_batch_p3_to_c3_dispatch!(
    YuvToRgbBatchP3ToC3,
    yuv_to_rgb_batch_p3_to_c3,
    yuv_to_rgb_batch_p3_to_c3,
    [u8 => yuv_to_rgb_batch_u8_p3_to_c3]
);
impl_color_convert_batch_p3_to_c3_dispatch!(
    YuvToRgbBatchP3ToC3Advanced,
    yuv_to_rgb_batch_p3_to_c3_advanced,
    yuv_to_rgb_batch_p3_to_c3_advanced,
    [u8 => yuv_to_rgb_batch_u8_p3_to_c3_advanced]
);
impl_color_convert_batch_p3_to_c3_dispatch!(
    YcbcrToRgbBatchP3ToC3,
    ycbcr_to_rgb_batch_p3_to_c3,
    ycbcr_to_rgb_batch_p3_to_c3,
    [u8 => ycbcr_to_rgb_batch_u8_p3_to_c3]
);
impl_color_convert_batch_p3_to_c3_dispatch!(
    YcbcrToRgbBatchP3ToC3Advanced,
    ycbcr_to_rgb_batch_p3_to_c3_advanced,
    ycbcr_to_rgb_batch_p3_to_c3_advanced,
    [u8 => ycbcr_to_rgb_batch_u8_p3_to_c3_advanced]
);
impl_color_convert_batch_p3_to_c3_dispatch!(
    YcbcrToBgrBatchP3ToC3,
    ycbcr_to_bgr_batch_p3_to_c3,
    ycbcr_to_bgr_batch_p3_to_c3,
    [u8 => ycbcr_to_bgr_batch_u8_p3_to_c3]
);
impl_color_convert_batch_p3_to_c3_dispatch!(
    YcbcrToBgrBatchP3ToC3Advanced,
    ycbcr_to_bgr_batch_p3_to_c3_advanced,
    ycbcr_to_bgr_batch_p3_to_c3_advanced,
    [u8 => ycbcr_to_bgr_batch_u8_p3_to_c3_advanced]
);
impl_color_convert_batch_p3_to_c3_dispatch!(
    YuvToBgrBatchP3ToC3,
    yuv_to_bgr_batch_p3_to_c3,
    yuv_to_bgr_batch_p3_to_c3,
    [u8 => yuv_to_bgr_batch_u8_p3_to_c3]
);
impl_color_convert_batch_p3_to_c3_dispatch!(
    YuvToBgrBatchP3ToC3Advanced,
    yuv_to_bgr_batch_p3_to_c3_advanced,
    yuv_to_bgr_batch_p3_to_c3_advanced,
    [u8 => yuv_to_bgr_batch_u8_p3_to_c3_advanced]
);

impl_subsampled_color_convert_batch_planar_to_packed!(
    yuv422_to_bgr_batch_u8_p3_to_c3,
    u8,
    C3,
    nppiYUV422ToBGRBatch_8u_P3C3R_Ctx,
    2,
    1
);
impl_subsampled_color_convert_batch_planar_to_packed_advanced!(
    yuv422_to_bgr_batch_u8_p3_to_c3_advanced,
    u8,
    C3,
    nppiYUV422ToBGRBatch_8u_P3C3R_Advanced_Ctx,
    2,
    1
);
impl_subsampled_color_convert_batch_planar_to_packed!(
    yuv420_to_bgr_batch_u8_p3_to_c3,
    u8,
    C3,
    nppiYUV420ToBGRBatch_8u_P3C3R_Ctx,
    2,
    2
);
impl_subsampled_color_convert_batch_planar_to_packed_advanced!(
    yuv420_to_bgr_batch_u8_p3_to_c3_advanced,
    u8,
    C3,
    nppiYUV420ToBGRBatch_8u_P3C3R_Advanced_Ctx,
    2,
    2
);

macro_rules! impl_subsampled_color_convert_batch_p3_to_c3_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source_plane_0: &[ImageView<'_, Self, C1>],
                source_plane_1: &[ImageView<'_, Self, C1>],
                source_plane_2: &[ImageView<'_, Self, C1>],
                destinations: &mut [ImageViewMut<'_, Self, C3>],
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source_plane_0: &[ImageView<'_, T, C1>],
            source_plane_1: &[ImageView<'_, T, C1>],
            source_plane_2: &[ImageView<'_, T, C1>],
            destinations: &mut [ImageViewMut<'_, T, C3>],
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(
                stream_context,
                source_plane_0,
                source_plane_1,
                source_plane_2,
                destinations,
            )
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source_plane_0: &[ImageView<'_, Self, C1>],
                    source_plane_1: &[ImageView<'_, Self, C1>],
                    source_plane_2: &[ImageView<'_, Self, C1>],
                    destinations: &mut [ImageViewMut<'_, Self, C3>],
                ) -> Result<()> {
                    $direct(
                        stream_context,
                        source_plane_0,
                        source_plane_1,
                        source_plane_2,
                        destinations,
                    )
                }
            }
        )*
    };
}

impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Yuv422ToRgbBatchP3ToC3,
    yuv422_to_rgb_batch_p3_to_c3,
    yuv422_to_rgb_batch_p3_to_c3,
    [u8 => yuv422_to_rgb_batch_u8_p3_to_c3]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Yuv422ToRgbBatchP3ToC3Advanced,
    yuv422_to_rgb_batch_p3_to_c3_advanced,
    yuv422_to_rgb_batch_p3_to_c3_advanced,
    [u8 => yuv422_to_rgb_batch_u8_p3_to_c3_advanced]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Yuv420ToRgbBatchP3ToC3,
    yuv420_to_rgb_batch_p3_to_c3,
    yuv420_to_rgb_batch_p3_to_c3,
    [u8 => yuv420_to_rgb_batch_u8_p3_to_c3]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Yuv420ToRgbBatchP3ToC3Advanced,
    yuv420_to_rgb_batch_p3_to_c3_advanced,
    yuv420_to_rgb_batch_p3_to_c3_advanced,
    [u8 => yuv420_to_rgb_batch_u8_p3_to_c3_advanced]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Ycbcr422ToRgbBatchP3ToC3,
    ycbcr422_to_rgb_batch_p3_to_c3,
    ycbcr422_to_rgb_batch_p3_to_c3,
    [u8 => ycbcr422_to_rgb_batch_u8_p3_to_c3]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Ycbcr422ToRgbBatchP3ToC3Advanced,
    ycbcr422_to_rgb_batch_p3_to_c3_advanced,
    ycbcr422_to_rgb_batch_p3_to_c3_advanced,
    [u8 => ycbcr422_to_rgb_batch_u8_p3_to_c3_advanced]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Ycbcr420ToRgbBatchP3ToC3,
    ycbcr420_to_rgb_batch_p3_to_c3,
    ycbcr420_to_rgb_batch_p3_to_c3,
    [u8 => ycbcr420_to_rgb_batch_u8_p3_to_c3]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Ycbcr420ToRgbBatchP3ToC3Advanced,
    ycbcr420_to_rgb_batch_p3_to_c3_advanced,
    ycbcr420_to_rgb_batch_p3_to_c3_advanced,
    [u8 => ycbcr420_to_rgb_batch_u8_p3_to_c3_advanced]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Ycbcr422ToBgrBatchP3ToC3,
    ycbcr422_to_bgr_batch_p3_to_c3,
    ycbcr422_to_bgr_batch_p3_to_c3,
    [u8 => ycbcr422_to_bgr_batch_u8_p3_to_c3]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Ycbcr422ToBgrBatchP3ToC3Advanced,
    ycbcr422_to_bgr_batch_p3_to_c3_advanced,
    ycbcr422_to_bgr_batch_p3_to_c3_advanced,
    [u8 => ycbcr422_to_bgr_batch_u8_p3_to_c3_advanced]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Ycbcr420ToBgrBatchP3ToC3,
    ycbcr420_to_bgr_batch_p3_to_c3,
    ycbcr420_to_bgr_batch_p3_to_c3,
    [u8 => ycbcr420_to_bgr_batch_u8_p3_to_c3]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Ycbcr420ToBgrBatchP3ToC3Advanced,
    ycbcr420_to_bgr_batch_p3_to_c3_advanced,
    ycbcr420_to_bgr_batch_p3_to_c3_advanced,
    [u8 => ycbcr420_to_bgr_batch_u8_p3_to_c3_advanced]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Yuv422ToBgrBatchP3ToC3,
    yuv422_to_bgr_batch_p3_to_c3,
    yuv422_to_bgr_batch_p3_to_c3,
    [u8 => yuv422_to_bgr_batch_u8_p3_to_c3]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Yuv422ToBgrBatchP3ToC3Advanced,
    yuv422_to_bgr_batch_p3_to_c3_advanced,
    yuv422_to_bgr_batch_p3_to_c3_advanced,
    [u8 => yuv422_to_bgr_batch_u8_p3_to_c3_advanced]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Yuv420ToBgrBatchP3ToC3,
    yuv420_to_bgr_batch_p3_to_c3,
    yuv420_to_bgr_batch_p3_to_c3,
    [u8 => yuv420_to_bgr_batch_u8_p3_to_c3]
);
impl_subsampled_color_convert_batch_p3_to_c3_dispatch!(
    Yuv420ToBgrBatchP3ToC3Advanced,
    yuv420_to_bgr_batch_p3_to_c3_advanced,
    yuv420_to_bgr_batch_p3_to_c3_advanced,
    [u8 => yuv420_to_bgr_batch_u8_p3_to_c3_advanced]
);

impl_color_convert_planar_to_planar!(yuv_to_bgr_u8_p3, u8, 3, nppiYUVToBGR_8u_P3R_Ctx);
impl_color_convert_packed_to_planar!(bgr_to_yuv_u8_c3_to_p3, u8, C3, 3, nppiBGRToYUV_8u_C3P3R_Ctx);
impl_color_convert_packed_to_planar!(
    bgr_to_yuv_u8_ac4_to_p4,
    u8,
    AC4,
    4,
    nppiBGRToYUV_8u_AC4P4R_Ctx
);
impl_color_convert_planar_to_packed!(yuv_to_bgr_u8_p3_to_c3, u8, 3, C3, nppiYUVToBGR_8u_P3C3R_Ctx);

macro_rules! impl_color_convert_packed_c3_to_planar_dispatch {
    ($trait:ident, $method:ident, $function:ident, $planes:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, C3>,
                destination: &mut PlanarImageViewMut<'_, Self, $planes>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, C3>,
            destination: &mut PlanarImageViewMut<'_, T, $planes>,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, source, destination)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, C3>,
                    destination: &mut PlanarImageViewMut<'_, Self, $planes>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )*
    };
}

macro_rules! impl_color_convert_planar_to_packed_c3_dispatch {
    ($trait:ident, $method:ident, $function:ident, $planes:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut ImageViewMut<'_, Self, C3>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, T, $planes>,
            destination: &mut ImageViewMut<'_, T, C3>,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, source, destination)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &PlanarImageView<'_, Self, $planes>,
                    destination: &mut ImageViewMut<'_, Self, C3>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )*
    };
}

macro_rules! impl_color_convert_planar_to_planar_dispatch {
    ($trait:ident, $method:ident, $function:ident, $planes:literal, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source: &PlanarImageView<'_, Self, $planes>,
                destination: &mut PlanarImageViewMut<'_, Self, $planes>,
            ) -> Result<()>;
        }

        pub fn $function<T>(
            stream_context: &StreamContext,
            source: &PlanarImageView<'_, T, $planes>,
            destination: &mut PlanarImageViewMut<'_, T, $planes>,
        ) -> Result<()>
        where
            T: $trait,
        {
            T::$method(stream_context, source, destination)
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &PlanarImageView<'_, Self, $planes>,
                    destination: &mut PlanarImageViewMut<'_, Self, $planes>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination)
                }
            }
        )*
    };
}

impl_color_convert_packed_c3_to_planar_dispatch!(
    RgbToYuvC3ToP3,
    rgb_to_yuv_c3_to_p3,
    rgb_to_yuv_c3_to_p3,
    3,
    [u8 => rgb_to_yuv_u8_c3_to_p3]
);
impl_color_convert_packed_c3_to_planar_dispatch!(
    BgrToYuvC3ToP3,
    bgr_to_yuv_c3_to_p3,
    bgr_to_yuv_c3_to_p3,
    3,
    [u8 => bgr_to_yuv_u8_c3_to_p3]
);
impl_color_convert_planar_to_packed_c3_dispatch!(
    YuvToRgbP3ToC3,
    yuv_to_rgb_p3_to_c3,
    yuv_to_rgb_p3_to_c3,
    3,
    [u8 => yuv_to_rgb_u8_p3_to_c3]
);
impl_color_convert_planar_to_packed_c3_dispatch!(
    YuvToBgrP3ToC3,
    yuv_to_bgr_p3_to_c3,
    yuv_to_bgr_p3_to_c3,
    3,
    [u8 => yuv_to_bgr_u8_p3_to_c3]
);
impl_color_convert_planar_to_planar_dispatch!(RgbToYuvP3, rgb_to_yuv_p3, rgb_to_yuv_p3, 3, [
    u8 => rgb_to_yuv_u8_p3,
]);
impl_color_convert_planar_to_planar_dispatch!(BgrToYuvP3, bgr_to_yuv_p3, bgr_to_yuv_p3, 3, [
    u8 => bgr_to_yuv_u8_p3,
]);
impl_color_convert_planar_to_planar_dispatch!(YuvToRgbP3, yuv_to_rgb_p3, yuv_to_rgb_p3, 3, [
    u8 => yuv_to_rgb_u8_p3,
]);
impl_color_convert_planar_to_planar_dispatch!(YuvToBgrP3, yuv_to_bgr_p3, yuv_to_bgr_p3, 3, [
    u8 => yuv_to_bgr_u8_p3,
]);
