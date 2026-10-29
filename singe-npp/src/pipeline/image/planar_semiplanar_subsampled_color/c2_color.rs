use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        memory::Image,
        view::{C1, C2, ImageOutputView, ImageView},
    },
    pipeline::image::planar::SemiplanarImage,
};

type SemiplanarToC2 = for<'source_y, 'source_uv, 'destination> unsafe fn(
    &StreamContext,
    &ImageView<'source_y, u8, C1>,
    &ImageView<'source_uv, u8, C2>,
    &mut ImageOutputView<'destination, u8, C2>,
) -> Result<()>;

impl SemiplanarImage<u8> {
    /// Converts semiplanar YCbCr 4:2:0 into a new packed YCbCr 4:2:2 image.
    ///
    /// The input has a full-resolution Y plane and an interleaved chroma plane with
    /// width divided by 2 and height divided by 2. Both planes use `u8` samples.
    /// The full-resolution dimensions must be divisible by the input and output subsampling
    /// factors; stored chroma dimensions must exactly match the declared sampling.
    /// Allocates a full-resolution C2 image containing the native YCbCr 4:2:2 packing.
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
    pub fn ycbcr420_to_ycbcr422_c2(self, stream_context: &StreamContext) -> Result<Image<u8, C2>>
    where
        u8: SupportedImage<C2>,
    {
        self.color_convert_to_c2(stream_context, color::ycbcr420_to_ycbcr422_u8_p2_to_c2)
    }

    /// Converts semiplanar YCbCr 4:2:0 into a new packed `CbYCr` 4:2:2 image.
    ///
    /// The input has a full-resolution Y plane and an interleaved chroma plane with
    /// width divided by 2 and height divided by 2. Both planes use `u8` samples.
    /// The full-resolution dimensions must be divisible by the input and output subsampling
    /// factors; stored chroma dimensions must exactly match the declared sampling.
    /// Allocates a full-resolution C2 image containing the native `CbYCr` 4:2:2 packing.
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
    pub fn ycbcr420_to_cbycr422_c2(self, stream_context: &StreamContext) -> Result<Image<u8, C2>>
    where
        u8: SupportedImage<C2>,
    {
        self.color_convert_to_c2(stream_context, color::ycbcr420_to_cbycr422_u8_p2_to_c2)
    }

    /// Converts semiplanar YCbCr 4:1:1 into a new packed YCbCr 4:2:2 image.
    ///
    /// The input has a full-resolution Y plane and an interleaved chroma plane with
    /// width divided by 4 and height divided by 1. Both planes use `u8` samples.
    /// The full-resolution dimensions must be divisible by the input and output subsampling
    /// factors; stored chroma dimensions must exactly match the declared sampling.
    /// Allocates a full-resolution C2 image containing the native YCbCr 4:2:2 packing.
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
    pub fn ycbcr411_to_ycbcr422_c2(self, stream_context: &StreamContext) -> Result<Image<u8, C2>>
    where
        u8: SupportedImage<C2>,
    {
        self.color_convert_to_c2(stream_context, color::ycbcr411_to_ycbcr422_u8_p2_to_c2)
    }

    fn color_convert_to_c2(
        self,
        stream_context: &StreamContext,
        operation: SemiplanarToC2,
    ) -> Result<Image<u8, C2>>
    where
        u8: SupportedImage<C2>,
    {
        self.prepare_eager(stream_context)?;

        let mut destination = Image::create_uninit(stream_context.cuda_context(), self.y.size())?;

        {
            let source_y = self.y.view()?;
            let source_uv = self.uv.view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe { operation(stream_context, &source_y, &source_uv, &mut destination_view) }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}
