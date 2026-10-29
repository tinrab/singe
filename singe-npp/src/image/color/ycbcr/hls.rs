use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color_descriptors::validate_same_size,
        view::{AC4, C3, ImageOutputView, ImageView, PlanarImageOutputView, PlanarImageView},
    },
    try_ffi,
    types::DataTypeLike,
};

impl_color_convert_packed_to_planar!(
    "Converts BGR color values to HLS color values.",
    bgr_to_hls_u8_c3_to_p3,
    u8,
    C3,
    3,
    nppiBGRToHLS_8u_C3P3R_Ctx
);

impl_color_convert_same_layout!(
    "Converts BGR color values to HLS color values. The destination alpha channel is not processed.",
    bgr_to_hls_u8_ac4,
    u8,
    AC4,
    nppiBGRToHLS_8u_AC4R_Ctx
);

impl_color_convert_packed_to_planar!(
    "Converts BGR color values to HLS color values.",
    bgr_to_hls_u8_ac4_to_p4,
    u8,
    AC4,
    4,
    nppiBGRToHLS_8u_AC4P4R_Ctx
);

impl_color_convert_planar_to_planar!(
    "Converts BGR color values to HLS color values.",
    bgr_to_hls_u8_p3,
    u8,
    3,
    nppiBGRToHLS_8u_P3R_Ctx
);

impl_color_convert_planar_to_planar!(
    "Converts BGR color values to HLS color values.",
    bgr_to_hls_u8_p4,
    u8,
    4,
    nppiBGRToHLS_8u_AP4R_Ctx
);

impl_color_convert_planar_to_packed!(
    "Converts BGR color values to HLS color values.",
    bgr_to_hls_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiBGRToHLS_8u_P3C3R_Ctx
);

impl_color_convert_planar_to_packed!(
    "Converts BGR color values to HLS color values. The destination alpha channel is not processed.",
    bgr_to_hls_u8_p4_to_ac4,
    u8,
    4,
    AC4,
    nppiBGRToHLS_8u_AP4C4R_Ctx
);

impl_color_convert_planar_to_packed!(
    "Converts HLS color values to BGR color values.",
    hls_to_bgr_u8_p3_to_c3,
    u8,
    3,
    C3,
    nppiHLSToBGR_8u_P3C3R_Ctx
);

impl_color_convert_planar_to_packed!(
    "Converts HLS color values to BGR color values. The destination alpha channel is not processed.",
    hls_to_bgr_u8_p4_to_ac4,
    u8,
    4,
    AC4,
    nppiHLSToBGR_8u_AP4C4R_Ctx
);

impl_color_convert_packed_to_planar!(
    "Converts HLS color values to BGR color values.",
    hls_to_bgr_u8_c3_to_p3,
    u8,
    C3,
    3,
    nppiHLSToBGR_8u_C3P3R_Ctx
);

impl_color_convert_packed_to_planar!(
    "Converts HLS color values to BGR color values.",
    hls_to_bgr_u8_ac4_to_p4,
    u8,
    AC4,
    4,
    nppiHLSToBGR_8u_AC4P4R_Ctx
);

impl_color_convert_planar_to_planar!(
    "Converts HLS color values to BGR color values.",
    hls_to_bgr_u8_p3,
    u8,
    3,
    nppiHLSToBGR_8u_P3R_Ctx
);

impl_color_convert_planar_to_planar!(
    "Converts HLS color values to BGR color values.",
    hls_to_bgr_u8_p4,
    u8,
    4,
    nppiHLSToBGR_8u_AP4R_Ctx
);

impl_color_convert_packed_c3_to_planar_dispatch!(
    "Converts BGR color values to HLS color values.",

    BgrToHlsC3ToP3,
    bgr_to_hls_c3_to_p3,
    bgr_to_hls_c3_to_p3,
    3,
    [u8 => bgr_to_hls_u8_c3_to_p3]

);

impl_color_convert_packed_c3_to_planar_dispatch!(
    "Converts HLS color values to BGR color values.",

    HlsToBgrC3ToP3,
    hls_to_bgr_c3_to_p3,
    hls_to_bgr_c3_to_p3,
    3,
    [u8 => hls_to_bgr_u8_c3_to_p3]

);

impl_color_convert_planar_to_packed_c3_dispatch!(
    "Converts BGR color values to HLS color values.",

    BgrToHlsP3ToC3,
    bgr_to_hls_p3_to_c3,
    bgr_to_hls_p3_to_c3,
    3,
    [u8 => bgr_to_hls_u8_p3_to_c3]

);

impl_color_convert_planar_to_packed_c3_dispatch!(
    "Converts HLS color values to BGR color values.",

    HlsToBgrP3ToC3,
    hls_to_bgr_p3_to_c3,
    hls_to_bgr_p3_to_c3,
    3,
    [u8 => hls_to_bgr_u8_p3_to_c3]

);

impl_color_convert_planar_to_planar_dispatch!(
    "Converts BGR color values to HLS color values.",
BgrToHlsP3, bgr_to_hls_p3, bgr_to_hls_p3, 3, [
    u8 => bgr_to_hls_u8_p3,
]
);

impl_color_convert_planar_to_planar_dispatch!(
    "Converts HLS color values to BGR color values.",
HlsToBgrP3, hls_to_bgr_p3, hls_to_bgr_p3, 3, [
    u8 => hls_to_bgr_u8_p3,
]
);
