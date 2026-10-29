use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        view::{C1, C2, ImageOutputView, ImageView},
    },
    pipeline::image::planar_pipeline_color::{ImagePipeline, PlanarImage},
};

type C2ToPlanarColorConvert = for<'source, 'r, 'g, 'b> unsafe fn(
    &StreamContext,
    &ImageView<'source, u8, C2>,
    &mut ImageOutputView<'r, u8, C1>,
    &mut ImageOutputView<'g, u8, C1>,
    &mut ImageOutputView<'b, u8, C1>,
) -> Result<()>;

impl ImagePipeline<'_, u8, C2>
where
    u8: SupportedImage<C1>,
{
    /// Converts packed YCbCr 4:2:2 into a new planar RGB image.
    ///
    /// The C2 input uses the native packed 4:2:2 component order named by the method.
    /// Its width must be even. Allocates three full-resolution C1 `u8` planes in R, G, B
    /// order; output dimensions match the source.
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
    pub fn ycbcr422_to_rgb_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::ycbcr422_to_rgb_c2_to_p3)
    }

    /// Converts packed `YCrCb` 4:2:2 into a new planar RGB image.
    ///
    /// The C2 input uses the native packed 4:2:2 component order named by the method.
    /// Its width must be even. Allocates three full-resolution C1 `u8` planes in R, G, B
    /// order; output dimensions match the source.
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
    pub fn ycrcb422_to_rgb_planar(self) -> Result<PlanarImage<u8, 3>> {
        self.color_convert_to_planar(color::ycrcb422_to_rgb_c2_to_p3)
    }

    fn color_convert_to_planar(
        self,
        operation: C2ToPlanarColorConvert,
    ) -> Result<PlanarImage<u8, 3>> {
        self.ensure_eager()?;

        let mut destination =
            PlanarImage::<u8, 3>::create_uninit(self.stream_context.cuda_context(), self.size())?;

        {
            let source = self.view()?;
            let [destination_r, destination_g, destination_b] = &mut destination.planes;
            let mut red_output = destination_r.output()?;
            let mut green_output = destination_g.output()?;
            let mut blue_output = destination_b.output()?;
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe {
                    operation(
                        self.stream_context,
                        &source,
                        &mut red_output,
                        &mut green_output,
                        &mut blue_output,
                    )
                }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}
