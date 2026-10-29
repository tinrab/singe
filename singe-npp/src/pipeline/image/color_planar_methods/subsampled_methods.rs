use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        view::{C1, C2, C3, ImageOutputView, ImageView},
    },
    pipeline::image::ImagePipeline,
};

mod source_methods;

impl ImagePipeline<'_, u8, C3>
where
    u8: SupportedImage<C3>,
{
    impl_static_planar3_to_planar2_convert!(
        "3 channel planar `YCbCr411` to 2 channel planar `YCbCr420` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr411_planar_to_ycbcr420_p2_into,
        color::ycbcr411_to_ycbcr420_u8_p3_to_p2
    );

    impl_static_planar3_to_planar3_convert!(
        "3 channel planar `YCbCr411` to 3 channel planar `YCbCr420` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr411_planar_to_ycbcr420_into,
        u8,
        color::ycbcr411_to_ycbcr420_u8_p3_to_p3
    );

    impl_static_planar_to_c2_convert!(
        "3 channel planar `YCbCr411` to 2 channel packed `YCbCr422` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr411_planar_to_ycbcr422_into,
        color::ycbcr411_to_ycbcr422_u8_p3_to_c2
    );

    impl_static_planar3_to_planar3_convert!(
        "3 channel planar `YCbCr411` to 3 channel planar `YCbCr422` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr411_planar_to_ycbcr422_p3_into,
        u8,
        color::ycbcr411_to_ycbcr422_u8_p3_to_p3
    );

    impl_static_planar_to_c2_convert!(
        "3 channel planar `YCbCr411` to 2 channel packed `YCrCb422` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr411_planar_to_ycrcb422_into,
        color::ycbcr411_to_ycrcb422_u8_p3_to_c2
    );

    impl_static_planar3_to_planar3_convert!(
        "3 channel planar `YCbCr411` to 3 channel planar `YCrCb422` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr411_planar_to_ycrcb422_p3_into,
        u8,
        color::ycbcr411_to_ycrcb422_u8_p3_to_p3
    );

    impl_static_planar3_to_planar2_convert!(
        "3 channel planar `YCbCr411` to 2 channel planar `YCbCr411` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr411_planar_to_p2_into,
        color::ycbcr411_u8_p3_to_p2
    );

    impl_static_planar3_to_planar2_convert!(
        "3 channel planar `YCbCr420` to 2 channel planar `YCbCr411` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr420_planar_to_ycbcr411_into,
        color::ycbcr420_to_ycbcr411_u8_p3_to_p2
    );

    impl_static_planar3_to_planar3_convert!(
        "3 channel planar `YCbCr420` to 3 channel planar `YCbCr422` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr420_planar_to_ycbcr422_into,
        u8,
        color::ycbcr420_to_ycbcr422_u8_p3_to_p3
    );

    impl_static_planar3_to_planar2_convert!(
        "3 channel planar `YCbCr420` to 2 channel planar `YCbCr420` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr420_planar_to_p2_into,
        color::ycbcr420_u8_p3_to_p2
    );

    impl_static_planar3_to_planar2_convert!(
        "3 channel planar `YCbCr422` to 2 channel planar `YCbCr411` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr422_planar_to_ycbcr411_p2_into,
        color::ycbcr422_to_ycbcr411_u8_p3_to_p2
    );

    impl_static_planar3_to_planar3_convert!(
        "3 channel planar `YCbCr422` to 3 channel planar `YCbCr411` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr422_planar_to_ycbcr411_into,
        u8,
        color::ycbcr422_to_ycbcr411_u8_p3_to_p3
    );

    impl_static_planar3_to_planar2_convert!(
        "3 channel planar `YCbCr422` to 2 channel planar `YCbCr420` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr422_planar_to_ycbcr420_p2_into,
        color::ycbcr422_to_ycbcr420_u8_p3_to_p2
    );

    impl_static_planar3_to_planar3_convert!(
        "3 channel planar `YCbCr422` to 3 channel planar `YCbCr420` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr422_planar_to_ycbcr420_into,
        u8,
        color::ycbcr422_to_ycbcr420_u8_p3_to_p3
    );

    impl_static_planar_to_c2_convert!(
        "3 channel planar `YCbCr422` to 2 channel packed `YCrCb422` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycbcr422_planar_to_ycrcb422_into,
        color::ycbcr422_to_ycrcb422_u8_p3_to_c2
    );

    impl_static_planar_to_c2_convert!(
        "3 channel planar `YCrCb420` to 2 channel packed `CbYCr422` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycrcb420_planar_to_cbycr422_into,
        color::ycrcb420_to_cbycr422_u8_p3_to_c2
    );

    impl_static_planar3_to_planar2_convert!(
        "3 channel planar `YCrCb420` to 2 channel planar `YCbCr411` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 4 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycrcb420_planar_to_ycbcr411_into,
        color::ycrcb420_to_ycbcr411_u8_p3_to_p2
    );

    impl_static_planar3_to_planar2_convert!(
        "3 channel planar `YCrCb420` to 2 channel planar `YCbCr420` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycrcb420_planar_to_ycbcr420_into,
        color::ycrcb420_to_ycbcr420_u8_p3_to_p2
    );

    impl_static_planar_to_c2_convert!(
        "3 channel planar `YCrCb420` to 2 channel packed `YCbCr422` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycrcb420_planar_to_ycbcr422_into,
        color::ycrcb420_to_ycbcr422_u8_p3_to_c2
    );

    impl_static_planar3_to_planar3_convert!(
        "3 channel planar `YCrCb420` to 3 channel planar `YCbCr422` sampling format conversion.\n\nChroma dimensions count chroma positions; interleaved chroma has two components per position.\nEach plane's byte stride is passed independently.\nRow padding remains unwritten.\n\nPrepares participating typed image and video planes on the selected stream, waiting for producers on other streams.\nWaits for the conversion to complete before returning successfully.\n\n# Errors\n\nReturns an error if CUDA context binding fails, any image plane belongs to another CUDA context, or NPP rejects the conversion parameters.\nSource and destination full-resolution dimensions must match.\nSource chroma width and height must equal the full-resolution dimensions divided by 2 and 2, respectively.\nSource full-resolution dimensions must be divisible by those subsampling factors.\nDestination chroma width and height must equal the full-resolution dimensions divided by 2 and 1, respectively.\nDestination full-resolution dimensions must be divisible by those subsampling factors.\nSeparate Cb and Cr planes must have equal sizes.\nReturns an error during stream capture, if preparing a region or waiting for its producer fails, or if stream synchronization fails.",
        ycrcb420_planar_to_ycbcr422_p3_into,
        u8,
        color::ycrcb420_to_ycbcr422_u8_p3_to_p3
    );
}
