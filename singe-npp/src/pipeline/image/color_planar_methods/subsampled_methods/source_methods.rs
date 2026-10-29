use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        view::{C1, C3, ImageOutputView, ImageView},
    },
    pipeline::image::ImagePipeline,
};

impl ImagePipeline<'_, u8, C3>
where
    u8: SupportedImage<C3>,
{
    impl_static_planar3_to_planar3_convert!(
        "3 channel planar RGB to 3 channel planar `YCbCr411` color conversion.\n\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        rgb_planar_to_ycbcr411_jpeg_into,
        u8,
        color::rgb_to_ycbcr411_jpeg_u8_p3
    );

    impl_static_planar3_to_planar3_convert!(
        "3 channel planar RGB to 3 channel planar `YCbCr420` color conversion.\n\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        rgb_planar_to_ycbcr420_jpeg_into,
        u8,
        color::rgb_to_ycbcr420_jpeg_u8_p3
    );

    impl_static_planar3_to_planar3_convert!(
        "3 channel planar RGB to 3 channel planar `YCbCr422` color conversion.\n\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        rgb_planar_to_ycbcr422_jpeg_into,
        u8,
        color::rgb_to_ycbcr422_jpeg_u8_p3
    );

    impl_static_planar3_to_planar3_convert!(
        "3 channel planar BGR to 3 channel planar `YCbCr411` color conversion.\n\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        bgr_planar_to_ycbcr411_jpeg_into,
        u8,
        color::bgr_to_ycbcr411_jpeg_u8_p3
    );

    impl_static_planar3_to_planar3_convert!(
        "3 channel planar BGR to 3 channel planar `YCbCr420` color conversion.\n\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        bgr_planar_to_ycbcr420_jpeg_into,
        u8,
        color::bgr_to_ycbcr420_jpeg_u8_p3
    );

    impl_static_planar3_to_planar3_convert!(
        "3 channel planar BGR to 3 channel planar `YCbCr422` color conversion.\n\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        bgr_planar_to_ycbcr422_jpeg_into,
        u8,
        color::bgr_to_ycbcr422_jpeg_u8_p3
    );

    impl_static_planar3_to_planar3_convert!(
        "3 channel planar RGB to 3 channel planar YUV420 color conversion.\n\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        rgb_planar_to_yuv420_into,
        u8,
        color::rgb_to_yuv420_u8_p3
    );

    impl_static_planar3_to_planar3_convert!(
        "3 channel planar RGB to 3 channel planar YUV422 color conversion.\n\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nChroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nFull-resolution dimensions must be divisible by those subsampling factors, and the Cb and Cr planes must have equal sizes.\nThe three full-resolution source color planes must have equal byte strides.\nChroma plane strides are passed independently.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        rgb_planar_to_yuv422_into,
        u8,
        color::rgb_to_yuv422_u8_p3
    );
}
