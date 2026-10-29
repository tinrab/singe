use crate::{
    context::StreamContext,
    error::Result,
    image::{SupportedImage, color, view::C1},
    pipeline::image::planar::{SemiplanarImage, SubsampledPlanarImage},
};

impl SemiplanarImage<u8> {
    /// Converts semiplanar YCbCr 4:1:1 into a new planar YCbCr 4:1:1 image.
    ///
    /// The input has a full-resolution Y plane and an interleaved chroma plane with
    /// width divided by 4 and height divided by 1. Both planes use `u8` samples.
    /// The full-resolution dimensions must be divisible by the input and output subsampling
    /// factors; stored chroma dimensions must exactly match the declared sampling.
    /// Allocates a full-resolution Y plane and two separate chroma planes of width divided
    /// by 4 and height divided by 1. The chroma storage order is Cb then Cr.
    /// Prepares source regions and waits for their producers on other streams. Rejects
    /// capture and returns after successful native conversion and stream completion.
    /// All logical output samples are initialized; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or source-region
    /// preparation, producer waits, checked subsampling geometry, plane allocation, view
    /// construction, native conversion, or synchronization fail. Input dimensions and
    /// component-plane sizes must satisfy the specified sampling; NPP can reject native
    /// conversion parameters.
    pub fn ycbcr411_to_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        u8: SupportedImage<C1>,
    {
        self.color_convert_to_subsampled_planar(stream_context, 4, 1, color::ycbcr411_u8_p2_to_p3)
    }

    /// Converts semiplanar YCbCr 4:1:1 into a new planar YCbCr 4:2:2 image.
    ///
    /// The input has a full-resolution Y plane and an interleaved chroma plane with
    /// width divided by 4 and height divided by 1. Both planes use `u8` samples.
    /// The full-resolution dimensions must be divisible by the input and output subsampling
    /// factors; stored chroma dimensions must exactly match the declared sampling.
    /// Allocates a full-resolution Y plane and two separate chroma planes of width divided
    /// by 2 and height divided by 1. The chroma storage order is Cb then Cr.
    /// Prepares source regions and waits for their producers on other streams. Rejects
    /// capture and returns after successful native conversion and stream completion.
    /// All logical output samples are initialized; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or source-region
    /// preparation, producer waits, checked subsampling geometry, plane allocation, view
    /// construction, native conversion, or synchronization fail. Input dimensions and
    /// component-plane sizes must satisfy the specified sampling; NPP can reject native
    /// conversion parameters.
    pub fn ycbcr411_to_ycbcr422_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        u8: SupportedImage<C1>,
    {
        self.color_convert_to_subsampled_planar(
            stream_context,
            2,
            1,
            color::ycbcr411_to_ycbcr422_u8_p2_to_p3,
        )
    }

    /// Converts semiplanar YCbCr 4:1:1 into a new planar YCbCr 4:2:0 image.
    ///
    /// The input has a full-resolution Y plane and an interleaved chroma plane with
    /// width divided by 4 and height divided by 1. Both planes use `u8` samples.
    /// The full-resolution dimensions must be divisible by the input and output subsampling
    /// factors; stored chroma dimensions must exactly match the declared sampling.
    /// Allocates a full-resolution Y plane and two separate chroma planes of width divided
    /// by 2 and height divided by 2. The chroma storage order is Cb then Cr.
    /// Prepares source regions and waits for their producers on other streams. Rejects
    /// capture and returns after successful native conversion and stream completion.
    /// All logical output samples are initialized; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or source-region
    /// preparation, producer waits, checked subsampling geometry, plane allocation, view
    /// construction, native conversion, or synchronization fail. Input dimensions and
    /// component-plane sizes must satisfy the specified sampling; NPP can reject native
    /// conversion parameters.
    pub fn ycbcr411_to_ycbcr420_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        u8: SupportedImage<C1>,
    {
        self.color_convert_to_subsampled_planar(
            stream_context,
            2,
            2,
            color::ycbcr411_to_ycbcr420_u8_p2_to_p3,
        )
    }

    /// Converts semiplanar YCbCr 4:1:1 into a new planar `YCrCb` 4:2:0 image.
    ///
    /// The input has a full-resolution Y plane and an interleaved chroma plane with
    /// width divided by 4 and height divided by 1. Both planes use `u8` samples.
    /// The full-resolution dimensions must be divisible by the input and output subsampling
    /// factors; stored chroma dimensions must exactly match the declared sampling.
    /// Allocates a full-resolution Y plane and two separate chroma planes of width divided
    /// by 2 and height divided by 2. The chroma storage order is Cr then Cb: `cb()`
    /// returns the first Cr plane and `cr()` the second Cb plane.
    /// Prepares source regions and waits for their producers on other streams. Rejects
    /// capture and returns after successful native conversion and stream completion.
    /// All logical output samples are initialized; row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or source-region
    /// preparation, producer waits, checked subsampling geometry, plane allocation, view
    /// construction, native conversion, or synchronization fail. Input dimensions and
    /// component-plane sizes must satisfy the specified sampling; NPP can reject native
    /// conversion parameters.
    pub fn ycbcr411_to_ycrcb420_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        u8: SupportedImage<C1>,
    {
        self.color_convert_to_subsampled_planar(
            stream_context,
            2,
            2,
            color::ycbcr411_to_ycrcb420_u8_p2_to_p3,
        )
    }
}
