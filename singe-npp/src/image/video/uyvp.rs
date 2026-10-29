use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, SupportedImage, color,
        video::Frame,
        view::{AC4, C3, ImageOutputView},
    },
    types::{ColorSpace, Point},
};

impl Frame<u8> {
    /// Decodes a packed UYVP frame and completes every RGB pixel before returning.
    /// Half-precision RGB channels use the normalized range `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native color conversion, or stream synchronization fail. The source
    /// frame must use `VideoFormat::Uyvp10`, and full-resolution RGB and frame dimensions
    /// must match. Packed UYVP requires an even pixel width. NPP can reject the selected
    /// color space or other conversion parameters.
    /// Image or video view construction can fail. Output image allocation and checked
    /// byte-size arithmetic can also fail.
    pub fn to_rgb<T>(
        &self,
        stream_context: &StreamContext,
        color_space: ColorSpace,
    ) -> Result<Image<T, C3>>
    where
        T: SupportedImage<C3> + color::Uyvp10uToRgb,
    {
        self.prepare_eager(stream_context)?;
        let mut destination =
            Image::<T, C3>::create_uninit(stream_context.cuda_context(), self.size())?;
        self.to_rgb_into(stream_context, &mut destination.output()?, color_space)?;
        // SAFETY: the completed native conversion initialized every RGB channel.
        Ok(unsafe { destination.assume_init() })
    }

    /// Decodes into caller-provided RGB storage and completes the conversion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native color conversion, or stream synchronization fail. The source
    /// frame must use `VideoFormat::Uyvp10`, and full-resolution RGB and frame dimensions
    /// must match. Packed UYVP requires an even pixel width. NPP can reject the selected
    /// color space or other conversion parameters.
    /// Image or video view construction can fail.
    pub fn to_rgb_into<T: color::Uyvp10uToRgb>(
        &self,
        stream_context: &StreamContext,
        destination: &mut ImageOutputView<'_, T, C3>,
        color_space: ColorSpace,
    ) -> Result<()> {
        let source = self.view()?;
        stream_context.sync_scope(|| {
            for plane in source.planes() {
                stream_context.prepare_region(plane)?;
            }
            stream_context.prepare_region(destination)?;
            // SAFETY: native dispatch checks packed geometry and initializes the full RGB ROI.
            // The completion scope retains every borrow through the queued accesses.
            unsafe {
                color::uyvp_10u_to_rgb(
                    stream_context,
                    &source,
                    Point { x: 0, y: 0 },
                    destination,
                    color_space,
                )
            }
        })
    }

    /// Decodes a packed UYVP frame, initializing all RGBA channels with the supplied alpha.
    /// Half-precision RGB channels use the normalized range `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native color conversion, or stream synchronization fail. The source
    /// frame must use `VideoFormat::Uyvp10`, and full-resolution RGB and frame dimensions
    /// must match. Packed UYVP requires an even pixel width. NPP can reject the selected
    /// color space or other conversion parameters.
    /// Image or video view construction can fail. Output image allocation and checked
    /// byte-size arithmetic can also fail.
    pub fn to_rgba<T>(
        &self,
        stream_context: &StreamContext,
        color_space: ColorSpace,
        alpha: T,
    ) -> Result<Image<T, AC4>>
    where
        T: SupportedImage<AC4> + color::Uyvp10uToRgba,
    {
        self.prepare_eager(stream_context)?;
        let mut destination =
            Image::<T, AC4>::create_uninit(stream_context.cuda_context(), self.size())?;
        self.to_rgba_into(
            stream_context,
            &mut destination.output()?,
            color_space,
            alpha,
        )?;
        // SAFETY: the completed native conversion initializes RGB and writes the explicit alpha.
        Ok(unsafe { destination.assume_init() })
    }

    /// Decodes into caller-provided RGBA storage and completes all four channels.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native color conversion, or stream synchronization fail. The source
    /// frame must use `VideoFormat::Uyvp10`, and full-resolution RGB and frame dimensions
    /// must match. Packed UYVP requires an even pixel width. NPP can reject the selected
    /// color space or other conversion parameters.
    /// Image or video view construction can fail.
    pub fn to_rgba_into<T: color::Uyvp10uToRgba>(
        &self,
        stream_context: &StreamContext,
        destination: &mut ImageOutputView<'_, T, AC4>,
        color_space: ColorSpace,
        alpha: T,
    ) -> Result<()> {
        let source = self.view()?;
        stream_context.sync_scope(|| {
            for plane in source.planes() {
                stream_context.prepare_region(plane)?;
            }
            stream_context.prepare_region(destination)?;
            // SAFETY: native dispatch checks packed geometry and writes every RGBA channel.
            // The completion scope retains every borrow through the queued accesses.
            unsafe {
                color::uyvp_10u_to_rgba(
                    stream_context,
                    &source,
                    Point { x: 0, y: 0 },
                    destination,
                    color_space,
                    alpha,
                )
            }
        })
    }
}
