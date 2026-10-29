use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        view::{C1, C2, ImageOutputView, ImageView},
    },
    pipeline::image::planar::{PlanarImage, SemiplanarImage},
};

type Nv12PlanarForward<T> = for<'r, 'g, 'b, 'y, 'uv> unsafe fn(
    &StreamContext,
    &ImageView<'r, T, C1>,
    &ImageView<'g, T, C1>,
    &ImageView<'b, T, C1>,
    &mut ImageOutputView<'y, T, C1>,
    &mut ImageOutputView<'uv, T, C2>,
    color::ColorTwistMatrix,
) -> Result<()>;

impl<T> PlanarImage<T, 3>
where
    T: Copy,
{
    pub(crate) fn color_twist_to_nv12(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
        operation: Nv12PlanarForward<T>,
    ) -> Result<SemiplanarImage<T>>
    where
        T: SupportedImage<C1> + SupportedImage<C2>,
    {
        self.prepare_eager(stream_context)?;

        let mut destination = SemiplanarImage::create_nv12_uninit(
            stream_context.cuda_context(),
            self.planes[0].size(),
        )?;

        {
            let source_r = self.planes[0].view()?;
            let source_g = self.planes[1].view()?;
            let source_b = self.planes[2].view()?;
            let mut destination_y = destination.y.output()?;
            let mut destination_uv = destination.uv.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe {
                    operation(
                        stream_context,
                        &source_r,
                        &source_g,
                        &source_b,
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
