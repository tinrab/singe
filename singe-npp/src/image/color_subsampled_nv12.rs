use super::*;

macro_rules! impl_nv12_color_twist_packed_inverse_dispatch {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source_y: &ImageView<'_, Self, C1>,
                source_uv: &ImageView<'_, Self, C2>,
                destination: &mut ImageViewMut<'_, Self, $layout>,
                twist: ColorTwistMatrix,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source_y: &ImageView<'_, Self, C1>,
                    source_uv: &ImageView<'_, Self, C2>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                    twist: ColorTwistMatrix,
                ) -> Result<()> {
                    $direct(stream_context, source_y, source_uv, destination, twist)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, T, C1>,
            source_uv: &ImageView<'_, T, C2>,
            destination: &mut ImageViewMut<'_, T, $layout>,
            twist: ColorTwistMatrix,
        ) -> Result<()> {
            T::$method(stream_context, source_y, source_uv, destination, twist)
        }
    };
}

macro_rules! impl_nv12_packed_inverse_dispatch {
    ($trait:ident, $method:ident, $function:ident, $layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source_y: &ImageView<'_, Self, C1>,
                source_uv: &ImageView<'_, Self, C2>,
                destination: &mut ImageViewMut<'_, Self, $layout>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source_y: &ImageView<'_, Self, C1>,
                    source_uv: &ImageView<'_, Self, C2>,
                    destination: &mut ImageViewMut<'_, Self, $layout>,
                ) -> Result<()> {
                    $direct(stream_context, source_y, source_uv, destination)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, T, C1>,
            source_uv: &ImageView<'_, T, C2>,
            destination: &mut ImageViewMut<'_, T, $layout>,
        ) -> Result<()> {
            T::$method(stream_context, source_y, source_uv, destination)
        }
    };
}

macro_rules! impl_packed_to_planar_subsampled_dispatch {
    ($trait:ident, $method:ident, $function:ident, $source_layout:ty, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source: &ImageView<'_, Self, $source_layout>,
                destination_y: &mut ImageViewMut<'_, Self, C1>,
                destination_cb: &mut ImageViewMut<'_, Self, C1>,
                destination_cr: &mut ImageViewMut<'_, Self, C1>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source: &ImageView<'_, Self, $source_layout>,
                    destination_y: &mut ImageViewMut<'_, Self, C1>,
                    destination_cb: &mut ImageViewMut<'_, Self, C1>,
                    destination_cr: &mut ImageViewMut<'_, Self, C1>,
                ) -> Result<()> {
                    $direct(stream_context, source, destination_y, destination_cb, destination_cr)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source: &ImageView<'_, T, $source_layout>,
            destination_y: &mut ImageViewMut<'_, T, C1>,
            destination_cb: &mut ImageViewMut<'_, T, C1>,
            destination_cr: &mut ImageViewMut<'_, T, C1>,
        ) -> Result<()> {
            T::$method(stream_context, source, destination_y, destination_cb, destination_cr)
        }
    };
}

macro_rules! impl_planar_to_packed_c2_subsampled_dispatch {
    ($trait:ident, $method:ident, $function:ident, [$($ty:ty => $direct:ident),* $(,)?]) => {
        pub trait $trait: DataTypeLike {
            fn $method(
                stream_context: &StreamContext,
                source_y: &ImageView<'_, Self, C1>,
                source_cb: &ImageView<'_, Self, C1>,
                source_cr: &ImageView<'_, Self, C1>,
                destination: &mut ImageViewMut<'_, Self, C2>,
            ) -> Result<()>;
        }

        $(
            impl $trait for $ty {
                fn $method(
                    stream_context: &StreamContext,
                    source_y: &ImageView<'_, Self, C1>,
                    source_cb: &ImageView<'_, Self, C1>,
                    source_cr: &ImageView<'_, Self, C1>,
                    destination: &mut ImageViewMut<'_, Self, C2>,
                ) -> Result<()> {
                    $direct(stream_context, source_y, source_cb, source_cr, destination)
                }
            }
        )*

        pub fn $function<T: $trait>(
            stream_context: &StreamContext,
            source_y: &ImageView<'_, T, C1>,
            source_cb: &ImageView<'_, T, C1>,
            source_cr: &ImageView<'_, T, C1>,
            destination: &mut ImageViewMut<'_, T, C2>,
        ) -> Result<()> {
            T::$method(stream_context, source_y, source_cb, source_cr, destination)
        }
    };
}

impl_nv12_color_twist_packed_inverse_dispatch!(
    Nv12ToRgbColorTwistP2ToC3,
    nv12_to_rgb_color_twist_p2_to_c3,
    nv12_to_rgb_color_twist_p2_to_c3,
    C3,
    [
        u8 => nv12_to_rgb_u8_color_twist_p2_to_c3,
        u16 => nv12_to_rgb_u16_color_twist_p2_to_c3
    ]
);

impl_nv12_packed_inverse_dispatch!(Nv12ToRgbP2ToC3, nv12_to_rgb_p2_to_c3, nv12_to_rgb_p2_to_c3, C3, [
    u8 => nv12_to_rgb_u8_p2_to_c3
]);
impl_nv12_packed_inverse_dispatch!(Nv12ToRgb709HdtvP2ToC3, nv12_to_rgb_709hdtv_p2_to_c3, nv12_to_rgb_709hdtv_p2_to_c3, C3, [
    u8 => nv12_to_rgb_709hdtv_u8_p2_to_c3
]);
impl_nv12_packed_inverse_dispatch!(Nv12ToRgb709CscP2ToC3, nv12_to_rgb_709csc_p2_to_c3, nv12_to_rgb_709csc_p2_to_c3, C3, [
    u8 => nv12_to_rgb_709csc_u8_p2_to_c3
]);
impl_nv12_packed_inverse_dispatch!(Nv12ToBgrP2ToC3, nv12_to_bgr_p2_to_c3, nv12_to_bgr_p2_to_c3, C3, [
    u8 => nv12_to_bgr_u8_p2_to_c3
]);
impl_nv12_packed_inverse_dispatch!(Nv12ToBgr709HdtvP2ToC3, nv12_to_bgr_709hdtv_p2_to_c3, nv12_to_bgr_709hdtv_p2_to_c3, C3, [
    u8 => nv12_to_bgr_709hdtv_u8_p2_to_c3
]);
impl_nv12_packed_inverse_dispatch!(Nv12ToBgr709CscP2ToC3, nv12_to_bgr_709csc_p2_to_c3, nv12_to_bgr_709csc_p2_to_c3, C3, [
    u8 => nv12_to_bgr_709csc_u8_p2_to_c3
]);
impl_nv12_packed_inverse_dispatch!(Nv21ToRgbP2ToC4, nv21_to_rgb_p2_to_c4, nv21_to_rgb_p2_to_c4, C4, [
    u8 => nv21_to_rgb_u8_p2_to_c4
]);
impl_nv12_packed_inverse_dispatch!(Nv21ToBgrP2ToC4, nv21_to_bgr_p2_to_c4, nv21_to_bgr_p2_to_c4, C4, [
    u8 => nv21_to_bgr_u8_p2_to_c4
]);

impl_packed_to_planar_subsampled_dispatch!(
    Ycbcr422C2ToP3,
    ycbcr422_c2_to_p3,
    ycbcr422_c2_to_p3,
    C2,
    [u8 => ycbcr422_u8_c2_to_p3]
);
impl_planar_to_packed_c2_subsampled_dispatch!(
    Ycbcr422P3ToC2,
    ycbcr422_p3_to_c2,
    ycbcr422_p3_to_c2,
    [u8 => ycbcr422_u8_p3_to_c2]
);
impl_planar_to_packed_c2_subsampled_dispatch!(
    Ycbcr422ToYcrcb422P3ToC2,
    ycbcr422_to_ycrcb422_p3_to_c2,
    ycbcr422_to_ycrcb422_p3_to_c2,
    [u8 => ycbcr422_to_ycrcb422_u8_p3_to_c2]
);
impl_planar_to_packed_c2_subsampled_dispatch!(
    Ycbcr411ToYcbcr422P3ToC2,
    ycbcr411_to_ycbcr422_p3_to_c2,
    ycbcr411_to_ycbcr422_p3_to_c2,
    [u8 => ycbcr411_to_ycbcr422_u8_p3_to_c2]
);
impl_planar_to_packed_c2_subsampled_dispatch!(
    Ycbcr411ToYcrcb422P3ToC2,
    ycbcr411_to_ycrcb422_p3_to_c2,
    ycbcr411_to_ycrcb422_p3_to_c2,
    [u8 => ycbcr411_to_ycrcb422_u8_p3_to_c2]
);
impl_planar_to_packed_c2_subsampled_dispatch!(
    RgbToYcbcr422P3ToC2,
    rgb_to_ycbcr422_p3_to_c2,
    rgb_to_ycbcr422_p3_to_c2,
    [u8 => rgb_to_ycbcr422_u8_p3_to_c2]
);
impl_planar_to_packed_c2_subsampled_dispatch!(
    RgbToYcrcb422P3ToC2,
    rgb_to_ycrcb422_p3_to_c2,
    rgb_to_ycrcb422_p3_to_c2,
    [u8 => rgb_to_ycrcb422_u8_p3_to_c2]
);
impl_packed_to_planar_subsampled_dispatch!(
    Ycbcr422ToRgbC2ToP3,
    ycbcr422_to_rgb_c2_to_p3,
    ycbcr422_to_rgb_c2_to_p3,
    C2,
    [u8 => ycbcr422_to_rgb_u8_c2_to_p3]
);
impl_packed_to_planar_subsampled_dispatch!(
    Ycrcb422ToRgbC2ToP3,
    ycrcb422_to_rgb_c2_to_p3,
    ycrcb422_to_rgb_c2_to_p3,
    C2,
    [u8 => ycrcb422_to_rgb_u8_c2_to_p3]
);
