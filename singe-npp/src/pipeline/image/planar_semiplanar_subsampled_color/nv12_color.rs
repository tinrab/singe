use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        view::{C1, C2, ImageOutputView, ImageView},
    },
    pipeline::image::planar::{SemiplanarImage, SubsampledPlanarImage},
};

type Nv12ToPlanar = for<'source_y, 'source_uv, 'y, 'u, 'v> unsafe fn(
    &StreamContext,
    &ImageView<'source_y, u8, C1>,
    &ImageView<'source_uv, u8, C2>,
    &mut ImageOutputView<'y, u8, C1>,
    &mut ImageOutputView<'u, u8, C1>,
    &mut ImageOutputView<'v, u8, C1>,
) -> Result<()>;

impl SemiplanarImage<u8> {
    /// Converts semiplanar NV12 into a new planar YUV 4:2:0 image.
    ///
    /// The input has a full-resolution Y plane and an interleaved chroma plane with
    /// width divided by 2 and height divided by 2. Both planes use `u8` samples.
    /// The full-resolution dimensions must be divisible by the input and output subsampling
    /// factors; stored chroma dimensions must exactly match the declared sampling.
    /// Allocates a full-resolution Y plane and two separate chroma planes of width divided
    /// by 2 and height divided by 2. The chroma storage order is U then V.
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
    pub fn nv12_to_yuv420_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        u8: SupportedImage<C1>,
    {
        self.color_convert_to_yuv420_planar(stream_context, color::nv12_to_yuv420_u8_p2_to_p3)
    }

    fn color_convert_to_yuv420_planar(
        self,
        stream_context: &StreamContext,
        operation: Nv12ToPlanar,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        u8: SupportedImage<C1>,
    {
        self.prepare_eager(stream_context)?;

        let mut destination = SubsampledPlanarImage::create_uninit(
            stream_context.cuda_context(),
            self.y.size(),
            2,
            2,
        )?;

        {
            let source_y = self.y.view()?;
            let source_uv = self.uv.view()?;
            let mut y = destination.y.output()?;
            let mut u = destination.cb.output()?;
            let mut v = destination.cr.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe {
                    operation(
                        stream_context,
                        &source_y,
                        &source_uv,
                        &mut y,
                        &mut u,
                        &mut v,
                    )
                }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}
