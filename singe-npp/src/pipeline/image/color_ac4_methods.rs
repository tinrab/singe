use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        view::{AC4, C2, ImageOutputView, ImageView, PlanarImageOutputView},
    },
    pipeline::image::ImagePipeline,
};

mod same_layout_methods;

impl ImagePipeline<'_, u8, AC4>
where
    u8: SupportedImage<AC4>,
{
    impl_static_different_layout_convert!(
        "Converts BGR color values to `CbYCr422` color values. Source alpha is ignored.\n\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        bgr_to_cbycr422_into,
        u8,
        AC4,
        C2,
        color::bgr_to_cbycr422_u8_ac4_to_c2
    );

    impl_static_different_layout_convert!(
        "Converts BGR color values to `CbYCr422` color values. Source alpha is ignored.\n\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        bgr_to_cbycr422_709hdtv_into,
        u8,
        AC4,
        C2,
        color::bgr_to_cbycr422_709hdtv_u8_ac4_to_c2
    );

    impl_static_different_layout_convert!(
        "Converts BGR color values to `YCbCr422` color values. Source alpha is ignored.\n\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        bgr_to_ycbcr422_into,
        u8,
        AC4,
        C2,
        color::bgr_to_ycbcr422_u8_ac4_to_c2
    );

    impl_static_packed_to_planar_convert!(
        "Converts RGB color values to YCbCr color values.\n\nPlane order follows the named color components.\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        rgb_to_ycbcr_planar_into,
        u8,
        AC4,
        3,
        color::rgb_to_ycbcr_u8_ac4_to_p3
    );

    impl_static_packed_to_planar_convert!(
        "Converts BGR color values to YCbCr color values.\n\nPlane order follows the named color components.\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        bgr_to_ycbcr_planar_into,
        u8,
        AC4,
        3,
        color::bgr_to_ycbcr_u8_ac4_to_p3
    );

    impl_static_packed_to_planar_convert!(
        "Converts BGR color values to YCbCr color values.\n\nPlane order follows the named color components.\nIgnored channels, pixels outside the processed ROI, and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails or a participating image belongs to another CUDA context.\nSource and destination image sizes must match.\nReturns an error if NPP rejects the image geometry or conversion parameters.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        bgr_to_ycbcr_planar4_into,
        u8,
        AC4,
        4,
        color::bgr_to_ycbcr_u8_ac4_to_p4
    );
}
