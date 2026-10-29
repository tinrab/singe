use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        view::{C1, C2, ImageOutputView, ImageView},
    },
    pipeline::image::ImagePipeline,
};

impl ImagePipeline<'_, u8, C2>
where
    u8: SupportedImage<C2>,
{
    impl_static_c2_to_planar_convert!(
        "Converts packed `YCbCr422` pixels to separate red, green, and blue destination planes.\n\nRow padding is not initialized.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the conversion parameters.\nAll source and destination views must have equal image sizes, and the three destination planes must have equal byte strides.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr422_to_rgb_planar_into,
        color::ycbcr422_to_rgb_c2_to_p3
    );

    impl_static_c2_to_planar_convert!(
        "Converts packed `YCrCb422` pixels to separate red, green, and blue destination planes.\n\nRow padding is not initialized.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image belongs to another CUDA context, or NPP rejects the conversion parameters.\nAll source and destination views must have equal image sizes, and the three destination planes must have equal byte strides.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycrcb422_to_rgb_planar_into,
        color::ycrcb422_to_rgb_c2_to_p3
    );

    impl_static_c2_to_planar3_convert!(
        "2 channel packed `CbYCr422` to 3 channel planar `YCbCr411` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        cbycr422_to_ycbcr411_into,
        color::cbycr422_to_ycbcr411_u8_c2_to_p3
    );

    impl_static_c2_to_planar2_convert!(
        "2 channel packed `CbYCr422` to 2 channel planar `YCbCr420` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        cbycr422_to_ycbcr420_p2_into,
        color::cbycr422_to_ycbcr420_u8_c2_to_p2
    );

    impl_static_c2_to_planar3_convert!(
        "2 channel packed `CbYCr422` to 3 channel planar `YCbCr420` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        cbycr422_to_ycbcr420_into,
        color::cbycr422_to_ycbcr420_u8_c2_to_p3
    );

    impl_static_c2_to_planar3_convert!(
        "2 channel packed `CbYCr422` to 3 channel planar `YCbCr422` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        cbycr422_to_ycbcr422_planar_into,
        color::cbycr422_to_ycbcr422_u8_c2_to_p3
    );

    impl_static_c2_to_planar3_convert!(
        "2 channel packed `CbYCr422` to 3 channel planar `YCrCb420` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        cbycr422_to_ycrcb420_into,
        color::cbycr422_to_ycrcb420_u8_c2_to_p3
    );

    impl_static_c2_to_planar2_convert!(
        "2 channel packed `YCbCr422` to 2 channel planar `YCbCr411` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr422_to_ycbcr411_p2_into,
        color::ycbcr422_to_ycbcr411_u8_c2_to_p2
    );

    impl_static_c2_to_planar3_convert!(
        "2 channel packed `YCbCr422` to 3 channel planar `YCbCr411` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr422_to_ycbcr411_into,
        color::ycbcr422_to_ycbcr411_u8_c2_to_p3
    );

    impl_static_c2_to_planar2_convert!(
        "2 channel packed `YCbCr422` to 2 channel planar `YCbCr420` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr422_to_ycbcr420_p2_into,
        color::ycbcr422_to_ycbcr420_u8_c2_to_p2
    );

    impl_static_c2_to_planar3_convert!(
        "2 channel packed `YCbCr422` to 3 channel planar `YCbCr420` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr422_to_ycbcr420_into,
        color::ycbcr422_to_ycbcr420_u8_c2_to_p3
    );

    impl_static_c2_to_planar3_convert!(
        "2 channel packed `YCbCr422` to 3 channel planar `YCrCb420` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr422_to_ycrcb420_into,
        color::ycbcr422_to_ycrcb420_u8_c2_to_p3
    );

    impl_static_c2_to_planar3_convert!(
        "2 channel packed `YCbCr422` to 3 channel planar `YCbCr422` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr422_to_planar_into,
        color::ycbcr422_u8_c2_to_p3
    );

    impl_static_c2_to_planar3_convert!(
        "2 channel packed `YCrCb422` to 3 channel planar `YCbCr411` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycrcb422_to_ycbcr411_into,
        color::ycrcb422_to_ycbcr411_u8_c2_to_p3
    );

    impl_static_c2_to_planar3_convert!(
        "2 channel packed `YCrCb422` to 3 channel planar `YCbCr420` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycrcb422_to_ycbcr420_into,
        color::ycrcb422_to_ycbcr420_u8_c2_to_p3
    );

    impl_static_c2_to_planar3_convert!(
        "2 channel packed `YCrCb422` to 3 channel planar `YCbCr422` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycrcb422_to_ycbcr422_into,
        color::ycrcb422_to_ycbcr422_u8_c2_to_p3
    );
}
