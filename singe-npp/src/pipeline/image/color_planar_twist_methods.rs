use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        view::{AC4, C1, C2, C3, C4, ImageOutputView, ImageView},
    },
    pipeline::image::{ColorTwistMatrix, ImagePipeline},
};

impl ImagePipeline<'_, u8, C3>
where
    u8: SupportedImage<C3>,
{
    impl_static_different_layout_twist!(
        "Three channel packed RGB conversion to two channel packed YUV 4:2:2, using a Color Twist to compute the exact color space arithmetic.\n\nIgnored channels and row padding may remain unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.\nEach source and destination image pair must have matching sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        rgb_to_yuv422_color_twist_into,
        u8,
        C3,
        C2,
        color::rgb_to_yuv422_u8_color_twist_c3_to_c2
    );

    impl_static_packed_to_planar3_twist!(
        "Three channel packed RGB conversion to three channel planar YUV 4:2:2, using a Color Twist to compute the exact color space arithmetic.\nPasses `twist` as a host color-twist matrix; coefficient values are not validated.\nChroma plane byte strides are passed independently.\nRow padding remains unwritten.\n\nChroma width and height divide the full-resolution dimensions by 2 and 1, respectively.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.\nFull-resolution source and destination dimensions must match.\nChroma planes must have matching dimensions and the exact size for the selected subsampling factors.\nFull-resolution dimensions must be divisible by those factors.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        rgb_to_yuv422_color_twist_planar_into,
        u8,
        C3,
        color::rgb_to_yuv422_u8_color_twist_c3_to_p3
    );

    impl_static_packed_to_planar3_twist!(
        "Three channel packed RGB conversion to three channel planar YUV 4:2:0, using a Color Twist to compute the exact color space arithmetic.\nPasses `twist` as a host color-twist matrix; coefficient values are not validated.\nChroma plane byte strides are passed independently.\nRow padding remains unwritten.\n\nChroma width and height divide the full-resolution dimensions by 2 and 2, respectively.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.\nFull-resolution source and destination dimensions must match.\nChroma planes must have matching dimensions and the exact size for the selected subsampling factors.\nFull-resolution dimensions must be divisible by those factors.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        rgb_to_yuv420_color_twist_planar_into,
        u8,
        C3,
        color::rgb_to_yuv420_u8_color_twist_c3_to_p3
    );

    impl_static_planar3_to_planar3_twist!(
        "Three channel planar RGB conversion to three channel planar YUV 4:2:2, using a Color Twist to compute the exact color space arithmetic.\nPasses `twist` as a host color-twist matrix; coefficient values are not validated.\nChroma plane byte strides are passed independently.\nRow padding remains unwritten.\n\nChroma width and height divide the full-resolution dimensions by 2 and 1, respectively.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.\nFull-resolution source and destination dimensions must match.\nChroma planes must have matching dimensions and the exact size for the selected subsampling factors.\nFull-resolution dimensions must be divisible by those factors.\nThe three full-resolution source color planes must have equal dimensions and byte strides.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        rgb_planar_to_yuv422_color_twist_into,
        u8,
        color::rgb_to_yuv422_u8_color_twist_p3
    );

    impl_static_planar3_to_planar3_twist!(
        "Three channel planar RGB conversion to three channel planar YUV 4:2:0, using a Color Twist to compute the exact color space arithmetic.\nPasses `twist` as a host color-twist matrix; coefficient values are not validated.\nChroma plane byte strides are passed independently.\nRow padding remains unwritten.\n\nChroma width and height divide the full-resolution dimensions by 2 and 2, respectively.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.\nFull-resolution source and destination dimensions must match.\nChroma planes must have matching dimensions and the exact size for the selected subsampling factors.\nFull-resolution dimensions must be divisible by those factors.\nThe three full-resolution source color planes must have equal dimensions and byte strides.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        rgb_planar_to_yuv420_color_twist_into,
        u8,
        color::rgb_to_yuv420_u8_color_twist_p3
    );

    impl_static_planar3_to_planar3_twist!(
        "Three channel planar YUV4:2:2 conversion to three channel planar RGB, using a Color Twist to compute the exact color space arithmetic.\nPasses `twist` as a host color-twist matrix; coefficient values are not validated.\nChroma plane byte strides are passed independently.\nRow padding remains unwritten.\n\nChroma width and height divide the full-resolution dimensions by 2 and 1, respectively.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.\nFull-resolution source and destination dimensions must match.\nChroma planes must have matching dimensions and the exact size for the selected subsampling factors.\nFull-resolution dimensions must be divisible by those factors.\nThe three full-resolution destination color planes must have equal dimensions and byte strides.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        yuv422_planar_to_rgb_color_twist_into,
        u8,
        color::yuv422_to_rgb_u8_color_twist_p3
    );

    impl_static_planar3_to_packed_twist!(
        "Three channel planar YUV4:2:2 conversion to three channel packed RGB, using a Color Twist to compute the exact color space arithmetic.\nPasses `twist` as a host color-twist matrix; coefficient values are not validated.\nChroma plane byte strides are passed independently.\nRow padding remains unwritten.\n\nChroma width and height divide the full-resolution dimensions by 2 and 1, respectively.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.\nFull-resolution source and destination dimensions must match.\nChroma planes must have matching dimensions and the exact size for the selected subsampling factors.\nFull-resolution dimensions must be divisible by those factors.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        yuv422_planar_to_rgb_color_twist_packed_into,
        u8,
        C3,
        color::yuv422_to_rgb_u8_color_twist_p3_to_c3
    );

    impl_static_planar3_to_ac4_twist!(
        "Three channel planar YUV4:2:2 conversion to four channel packed RGB, using a Color Twist to compute the exact color space arithmetic, with user set alpha.\nPasses `twist` as a host color-twist matrix; coefficient values are not validated.\nChroma plane byte strides are passed independently.\nRow padding remains unwritten.\n\nChroma width and height divide the full-resolution dimensions by 2 and 1, respectively.\nWrites `alpha` to the fourth destination channel.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.\nFull-resolution source and destination dimensions must match.\nChroma planes must have matching dimensions and the exact size for the selected subsampling factors.\nFull-resolution dimensions must be divisible by those factors.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        yuv422_planar_to_rgba_color_twist_into,
        u8,
        color::yuv422_to_rgb_u8_color_twist_p3_to_ac4
    );

    impl_static_planar3_to_planar3_twist!(
        "Three channel planar YUV4:2:0 conversion to three channel planar RGB, using a Color Twist to compute the exact color space arithmetic.\nPasses `twist` as a host color-twist matrix; coefficient values are not validated.\nChroma plane byte strides are passed independently.\nRow padding remains unwritten.\n\nChroma width and height divide the full-resolution dimensions by 2 and 2, respectively.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.\nFull-resolution source and destination dimensions must match.\nChroma planes must have matching dimensions and the exact size for the selected subsampling factors.\nFull-resolution dimensions must be divisible by those factors.\nThe three full-resolution destination color planes must have equal dimensions and byte strides.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        yuv420_planar_to_rgb_color_twist_into,
        u8,
        color::yuv420_to_rgb_u8_color_twist_p3
    );

    impl_static_planar3_to_packed_twist!(
        "Three channel planar YUV4:2:0 conversion to three channel packed RGB, using a Color Twist to compute the exact color space arithmetic.\nPasses `twist` as a host color-twist matrix; coefficient values are not validated.\nChroma plane byte strides are passed independently.\nRow padding remains unwritten.\n\nChroma width and height divide the full-resolution dimensions by 2 and 2, respectively.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.\nFull-resolution source and destination dimensions must match.\nChroma planes must have matching dimensions and the exact size for the selected subsampling factors.\nFull-resolution dimensions must be divisible by those factors.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        yuv420_planar_to_rgb_color_twist_packed_into,
        u8,
        C3,
        color::yuv420_to_rgb_u8_color_twist_p3_to_c3
    );

    impl_static_planar3_to_packed_twist!(
        "Three channel planar YUV4:2:0 conversion to four channel packed RGB, using a Color Twist to compute the exact color space arithmetic, with constant alpha (0xFF).\nPasses `twist` as a host color-twist matrix; coefficient values are not validated.\nChroma plane byte strides are passed independently.\nRow padding remains unwritten.\n\nChroma width and height divide the full-resolution dimensions by 2 and 2, respectively.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.\nFull-resolution source and destination dimensions must match.\nChroma planes must have matching dimensions and the exact size for the selected subsampling factors.\nFull-resolution dimensions must be divisible by those factors.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        yuv420_planar_to_rgb_color_twist_c4_into,
        u8,
        C4,
        color::yuv420_to_rgb_u8_color_twist_p3_to_c4
    );

    impl_static_planar3_to_ac4_twist!(
        "Three channel planar YUV4:2:0 conversion to four channel packed RGB, using a Color Twist to compute the exact color space arithmetic, with user set alpha.\nPasses `twist` as a host color-twist matrix; coefficient values are not validated.\nChroma plane byte strides are passed independently.\nRow padding remains unwritten.\n\nChroma width and height divide the full-resolution dimensions by 2 and 2, respectively.\nWrites `alpha` to the fourth destination channel.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion.\nFull-resolution source and destination dimensions must match.\nChroma planes must have matching dimensions and the exact size for the selected subsampling factors.\nFull-resolution dimensions must be divisible by those factors.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        yuv420_planar_to_rgba_color_twist_into,
        u8,
        color::yuv420_to_rgb_u8_color_twist_p3_to_ac4
    );
}
