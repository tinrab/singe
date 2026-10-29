use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        view::{
            C1, C2, C3, C4, ImageOutputView, ImageView, PlanarImageOutputView, PlanarImageView,
        },
    },
    pipeline::image::ImagePipeline,
    types::{ColorSpace, Point},
};

mod subsampled_methods;

impl ImagePipeline<'_, u8, C3>
where
    u8: SupportedImage<C3>,
{
    impl_static_packed_to_planar_convert!(
        "Converts RGB color values to YCbCr color values.\n\nPlane order follows the named color components.\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        rgb_to_ycbcr_planar_into,
        u8,
        C3,
        3,
        color::rgb_to_ycbcr_u8_c3_to_p3
    );

    impl_static_packed_to_planar_convert!(
        "Converts BGR color values to YCbCr color values.\n\nPlane order follows the named color components.\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        bgr_to_ycbcr_planar_into,
        u8,
        C3,
        3,
        color::bgr_to_ycbcr_u8_c3_to_p3
    );

    impl_static_packed_to_planar_convert!(
        "Converts BGR color values to YUV color values.\n\nPlane order follows the named color components.\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        bgr_to_yuv_planar_into,
        u8,
        C3,
        3,
        color::bgr_to_yuv_u8_c3_to_p3
    );

    impl_static_packed_to_planar_convert!(
        "Converts BGR color values to HLS color values.\n\nPlane order follows the named color components.\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        bgr_to_hls_planar_into,
        u8,
        C3,
        3,
        color::bgr_to_hls_u8_c3_to_p3
    );

    impl_static_planar_to_packed_convert!(
        "Converts BGR color values to HLS color values.\n\nPlane order follows the named color components.\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        bgr_planar_to_hls_into,
        u8,
        3,
        C3,
        color::bgr_to_hls_u8_p3_to_c3
    );

    impl_static_planar_to_packed_convert!(
        "Converts HLS color values to BGR color values.\n\nPlane order follows the named color components.\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        hls_planar_to_bgr_into,
        u8,
        3,
        C3,
        color::hls_to_bgr_u8_p3_to_c3
    );

    impl_static_planar_to_packed_convert!(
        "Converts YCbCr color values to BGR color values.\n\nPlane order follows the named color components.\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr_planar_to_bgr_709csc_into,
        u8,
        3,
        C3,
        color::ycbcr_to_bgr_709csc_u8_p3_to_c3
    );

    impl_static_planar_to_packed_constant_alpha_convert!(
        "Converts YCbCr color values to RGB color values. Writes `alpha` to every destination alpha component.\n\nPlane order follows the named color components.\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr_planar_to_rgb_c4_into,
        u8,
        3,
        color::ycbcr_to_rgb_u8_p3_to_c4
    );

    impl_static_planar_to_packed_constant_alpha_convert!(
        "Converts YCbCr color values to BGR color values. Writes `alpha` to every destination alpha component.\n\nPlane order follows the named color components.\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr_planar_to_bgr_c4_into,
        u8,
        3,
        color::ycbcr_to_bgr_u8_p3_to_c4
    );

    impl_static_planar_to_packed_constant_alpha_convert!(
        "Converts YCbCr color values to BGR color values. Writes `alpha` to every destination alpha component.\n\nPlane order follows the named color components.\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr_planar_to_bgr_709csc_c4_into,
        u8,
        3,
        color::ycbcr_to_bgr_709csc_u8_p3_to_c4
    );

    impl_static_planar_to_c2_convert!(
        "Converts the separate red, green, and blue source planes to packed `YCbCr422` pixels.\n\nRow padding is not initialized.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the conversion parameters.\nAll source and destination views must have equal image sizes, and the three source planes must have equal byte strides.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        rgb_planar_to_ycbcr422_into,
        color::rgb_to_ycbcr422_p3_to_c2
    );

    impl_static_rgb_to_uyvp_planar!(
        "Three unsigned 8-bit packed RGB channels color space conversion to three channel unsigned planar 10-bit UYVP pixel.\n`source_offset` must be `(0, 0)`.\n`color_space` selects the native color-conversion standard.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, a typed image or video plane belongs to another CUDA context, `source_offset` is nonzero, or NPP rejects the conversion.\n`destination_step` must be positive.\nRaw destination pointers are not checked for context, allocation bounds, or overlap.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.\n\n# Safety\n\nEvery raw destination must cover the native planar UYVP layout and pitch for the source ROI in the selected CUDA context.\nDestinations must be disjoint from the source and one another, and exclusively accessible until this call returns.\nEstablish access ordering for these untracked destinations, including previously queued accesses on other streams.\nNative pixel packing, component ranges, byte strides, alignment, and allocation bounds must be valid.",
        rgb_to_uyvp_10u_planes_into,
        u8,
        C3,
        color::rgb_to_uyvp_10u_u8_c3_to_p3
    );

    impl_static_rgb_planar_to_uyvp!(
        "Three unsigned 8-bit planar RGB channels color space conversion to three channel unsigned packed 10-bit UYVP pixel.\n`source_offset` must be `(0, 0)`.\n`color_space` selects the native color-conversion standard.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, a typed image or video plane belongs to another CUDA context, `source_offset` is nonzero, or NPP rejects the conversion.\nThe video view must use `VideoFormat::Uyvp10`, and full-resolution source and destination dimensions must match.\nThe three RGB source planes must have equal dimensions and byte strides.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        rgb_planar_to_uyvp_10u_into,
        u8,
        color::rgb_to_uyvp_10u_u8_p3
    );

    impl_static_uyvp_to_rgb_planar!(
        "Three channel unsigned packed 10-bit UYVP pixel color space conversion to three unsigned planar 8-bit channels RGB.\n`source_offset` must be `(0, 0)`.\n`color_space` selects the native color-conversion standard.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, a typed image or video plane belongs to another CUDA context, `source_offset` is nonzero, or NPP rejects the conversion.\nThe video view must use `VideoFormat::Uyvp10`, and full-resolution source and destination dimensions must match.\nThe three RGB destination planes must have equal dimensions and byte strides.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        uyvp_10u_to_rgb_planar_into,
        u8,
        color::uyvp_10u_to_rgb_u8_p3
    );
}
