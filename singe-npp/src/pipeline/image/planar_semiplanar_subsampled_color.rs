use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        view::{C1, C2, ImageOutputView, ImageView},
    },
    pipeline::image::planar::{SemiplanarImage, SubsampledPlanarImage},
};

mod c2_color;
mod nv12_color;
mod ycbcr411_color;

type SemiplanarToSubsampledPlanar = for<'source_y, 'source_uv, 'y, 'cb, 'cr> unsafe fn(
    &StreamContext,
    &ImageView<'source_y, u8, C1>,
    &ImageView<'source_uv, u8, C2>,
    &mut ImageOutputView<'y, u8, C1>,
    &mut ImageOutputView<'cb, u8, C1>,
    &mut ImageOutputView<'cr, u8, C1>,
)
    -> Result<()>;

impl SemiplanarImage<u8> {
    /// Converts semiplanar YCbCr 4:2:0 into a new planar YCbCr 4:2:0 image.
    ///
    /// The input has a full-resolution Y plane and an interleaved chroma plane with
    /// width divided by 2 and height divided by 2. Both planes use `u8` samples.
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
    pub fn ycbcr420_to_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        u8: SupportedImage<C1>,
    {
        self.color_convert_to_subsampled_planar(stream_context, 2, 2, color::ycbcr420_u8_p2_to_p3)
    }

    /// Converts semiplanar YCbCr 4:2:0 into a new planar YCbCr 4:2:2 image.
    ///
    /// The input has a full-resolution Y plane and an interleaved chroma plane with
    /// width divided by 2 and height divided by 2. Both planes use `u8` samples.
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
    pub fn ycbcr420_to_ycbcr422_planar(
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
            color::ycbcr420_to_ycbcr422_u8_p2_to_p3,
        )
    }

    /// Converts semiplanar YCbCr 4:2:0 into a new planar YCbCr 4:1:1 image.
    ///
    /// The input has a full-resolution Y plane and an interleaved chroma plane with
    /// width divided by 2 and height divided by 2. Both planes use `u8` samples.
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
    pub fn ycbcr420_to_ycbcr411_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        u8: SupportedImage<C1>,
    {
        self.color_convert_to_subsampled_planar(
            stream_context,
            4,
            1,
            color::ycbcr420_to_ycbcr411_u8_p2_to_p3,
        )
    }

    /// Converts semiplanar YCbCr 4:2:0 into a new planar `YCrCb` 4:2:0 image.
    ///
    /// The input has a full-resolution Y plane and an interleaved chroma plane with
    /// width divided by 2 and height divided by 2. Both planes use `u8` samples.
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
    pub fn ycbcr420_to_ycrcb420_planar(
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
            color::ycbcr420_to_ycrcb420_u8_p2_to_p3,
        )
    }

    fn color_convert_to_subsampled_planar(
        self,
        stream_context: &StreamContext,
        horizontal_subsampling: i32,
        vertical_subsampling: i32,
        operation: SemiplanarToSubsampledPlanar,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        u8: SupportedImage<C1>,
    {
        self.prepare_eager(stream_context)?;

        let mut destination = SubsampledPlanarImage::create_uninit(
            stream_context.cuda_context(),
            self.y.size(),
            horizontal_subsampling,
            vertical_subsampling,
        )?;

        {
            let source_y = self.y.view()?;
            let source_uv = self.uv.view()?;
            let mut y = destination.y.output()?;
            let mut cb = destination.cb.output()?;
            let mut cr = destination.cr.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe {
                    operation(
                        stream_context,
                        &source_y,
                        &source_uv,
                        &mut y,
                        &mut cb,
                        &mut cr,
                    )
                }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}
