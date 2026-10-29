use crate::{
    error::Result,
    image::{
        SupportedImage, color,
        view::{C1, C2, C3},
    },
    pipeline::image::{
        planar::SemiplanarImage,
        planar_pipeline_color::subsampled_color::{ImagePipeline, Nv12PackedForward},
    },
};

impl ImagePipeline<'_, u8, C3>
where
    u8: SupportedImage<C1>,
{
    /// Converts packed RGB into a new NV12-style semiplanar image using `twist`.
    ///
    /// Allocates a full-resolution Y plane and an interleaved U/V C2 plane at half width
    /// and half height. Width and height must be even. Retains the `u8` sample representation.
    /// The host three-by-four `f32` twist matrix determines the native RGB-to-YUV transform
    /// and is forwarded without coefficient validation.
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
    pub fn rgb_to_nv12(self, twist: color::ColorTwistMatrix) -> Result<SemiplanarImage<u8>>
    where
        u8: SupportedImage<C2>,
    {
        self.color_twist_to_nv12(twist, color::rgb_to_nv12_u8_color_twist_c3_to_p2)
    }

    fn color_twist_to_nv12(
        self,
        twist: color::ColorTwistMatrix,
        operation: Nv12PackedForward<u8, C3>,
    ) -> Result<SemiplanarImage<u8>>
    where
        u8: SupportedImage<C2>,
    {
        self.ensure_eager()?;

        let mut destination =
            SemiplanarImage::create_nv12_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let mut destination_y = destination.y.output()?;
            let mut destination_uv = destination.uv.output()?;
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe {
                    operation(
                        self.stream_context,
                        &source,
                        &mut destination_y,
                        &mut destination_uv,
                        twist,
                    )
                }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}
