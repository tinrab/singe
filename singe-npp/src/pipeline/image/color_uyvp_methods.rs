use crate::{
    context::StreamContext,
    error::Result,
    image::{
        color,
        video::{Frame, VideoFormat, VideoOutputView},
        view::{AC4, C3, ImageView},
    },
    pipeline::image::ImagePipeline,
    types::{ColorSpace, Point},
};

mod u16_methods;

impl<T: color::RgbToUyvp10u> ImagePipeline<'_, T, C3> {
    /// Converts every source pixel into packed UYVP and completes the conversion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native color conversion, or stream synchronization fail. The destination
    /// frame must use `VideoFormat::Uyvp10`, and full-resolution RGB and frame dimensions
    /// must match. Packed UYVP requires an even pixel width. NPP can reject the selected
    /// color space or other conversion parameters.
    /// Image or video view construction can fail.
    pub fn rgb_to_uyvp_10u_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        destination: &mut VideoOutputView<'_, u8>,
        color_space: ColorSpace,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        for plane in destination.planes() {
            stream_context.prepare_region(plane)?;
        }
        stream_context.sync_scope(|| {
            // SAFETY: checked native dispatch writes the packed extent, and the scope retains
            // the exclusive destination and source until queued accesses have completed.
            unsafe {
                color::rgb_to_uyvp_10u(
                    stream_context,
                    source,
                    Point { x: 0, y: 0 },
                    destination,
                    color_space,
                )
            }
        })
    }

    /// Produces a completed packed UYVP frame with five stored bytes per two pixels.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native color conversion, or stream synchronization fail. The destination
    /// frame must use `VideoFormat::Uyvp10`, and full-resolution RGB and frame dimensions
    /// must match. Packed UYVP requires an even pixel width. NPP can reject the selected
    /// color space or other conversion parameters.
    /// Image or video view construction can fail. Packed frame allocation and checked
    /// byte-size arithmetic can also fail.
    pub fn rgb_to_uyvp_10u(self, color_space: ColorSpace) -> Result<Frame<u8>> {
        self.ensure_eager()?;
        let mut destination = Frame::<u8>::create_uninit(
            self.stream_context.cuda_context(),
            VideoFormat::Uyvp10,
            self.size(),
        )?;
        Self::rgb_to_uyvp_10u_into(
            self.stream_context,
            &self.view()?,
            &mut destination.output()?,
            color_space,
        )?;
        // SAFETY: successful completed dispatch writes every stored byte of the packed frame.
        Ok(unsafe { destination.assume_init() })
    }
}

impl<T: color::RgbaToUyvp10u> ImagePipeline<'_, T, AC4> {
    /// Converts every source pixel into packed UYVP and completes the conversion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native color conversion, or stream synchronization fail. The destination
    /// frame must use `VideoFormat::Uyvp10`, and full-resolution RGB and frame dimensions
    /// must match. Packed UYVP requires an even pixel width. NPP can reject the selected
    /// color space or other conversion parameters.
    /// Image or video view construction can fail.
    pub fn rgba_to_uyvp_10u_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, AC4>,
        destination: &mut VideoOutputView<'_, u8>,
        color_space: ColorSpace,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        stream_context.prepare_region(source)?;
        for plane in destination.planes() {
            stream_context.prepare_region(plane)?;
        }
        stream_context.sync_scope(|| {
            // SAFETY: checked native dispatch writes the packed extent, and the scope retains
            // the exclusive destination and source until queued accesses have completed.
            unsafe {
                color::rgba_to_uyvp_10u(
                    stream_context,
                    source,
                    Point { x: 0, y: 0 },
                    destination,
                    color_space,
                )
            }
        })
    }

    /// Produces a completed packed UYVP frame, discarding source alpha.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native color conversion, or stream synchronization fail. The destination
    /// frame must use `VideoFormat::Uyvp10`, and full-resolution RGB and frame dimensions
    /// must match. Packed UYVP requires an even pixel width. NPP can reject the selected
    /// color space or other conversion parameters.
    /// Image or video view construction can fail. Packed frame allocation and checked
    /// byte-size arithmetic can also fail.
    pub fn rgba_to_uyvp_10u(self, color_space: ColorSpace) -> Result<Frame<u8>> {
        self.ensure_eager()?;
        let mut destination = Frame::<u8>::create_uninit(
            self.stream_context.cuda_context(),
            VideoFormat::Uyvp10,
            self.size(),
        )?;
        Self::rgba_to_uyvp_10u_into(
            self.stream_context,
            &self.view()?,
            &mut destination.output()?,
            color_space,
        )?;
        // SAFETY: successful completed dispatch writes every stored byte of the packed frame.
        Ok(unsafe { destination.assume_init() })
    }
}
