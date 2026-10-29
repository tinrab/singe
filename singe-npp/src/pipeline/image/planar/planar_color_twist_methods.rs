use crate::{
    context::StreamContext,
    error::Result,
    image::{
        SupportedImage, color,
        view::{C2, PlanarImageOutputView, PlanarImageView},
    },
    pipeline::image::planar::{PlanarImage, SemiplanarImage},
};

type PlanarColorTwist<T> = for<'source, 'destination> unsafe fn(
    &StreamContext,
    &PlanarImageView<'source, T, 3>,
    &mut PlanarImageOutputView<'destination, T, 3>,
    color::ColorTwistMatrix,
) -> Result<()>;

impl PlanarImage<f32, 3> {
    /// 3 channel 32-bit floating point planar color twist.
    ///
    /// Uses [`color_twist_f32_p3`](crate::image::raw::color::color_twist_f32_p3).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    /// Each source and destination image pair must have matching sizes.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn color_twist(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<Self> {
        self.color_twist_with(stream_context, twist, color::color_twist_f32_p3)
    }

    fn color_twist_with(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
        operation: PlanarColorTwist<f32>,
    ) -> Result<Self> {
        self.prepare_eager(stream_context)?;

        let mut destination =
            Self::create_uninit(stream_context.cuda_context(), self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe { operation(stream_context, &source, &mut destination_view, twist) }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}

impl PlanarImage<u16, 3> {
    /// 3 channel planar color twist.
    ///
    /// Uses [`color_twist_u16_p3`](crate::image::raw::color::color_twist_u16_p3).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    /// Each source and destination image pair must have matching sizes.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn color_twist(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<Self> {
        self.color_twist_with(stream_context, twist, color::color_twist_u16_p3)
    }

    /// Three channel planar RGB conversion to two channel planar NV12, using a Color Twist to compute the exact color space arithmetic.
    /// Chroma dimensions count positions; each interleaved chroma position stores two components.
    /// Passes `twist` as a host color-twist matrix; coefficient values are not validated.
    /// Destination plane strides are passed independently.
    ///
    /// Uses [`rgb_to_nv12_u16_color_twist_p3`](crate::image::raw::color::rgb_to_nv12_u16_color_twist_p3).
    /// Allocates a full-resolution Y plane and an interleaved UV plane with half the width
    /// and half the height. Both full-resolution dimensions must be even.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, any participating image plane belongs to another CUDA context, or NPP rejects the conversion parameters.
    /// The full-resolution source and destination dimensions must match.
    /// Full-resolution width and height must be even; each chroma plane must have half the full-resolution width and height.
    /// The three full-resolution source color planes must have equal dimensions and byte strides.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn rgb_to_nv12(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<SemiplanarImage<u16>>
    where
        u16: SupportedImage<C2>,
    {
        self.color_twist_to_nv12(stream_context, twist, color::rgb_to_nv12_u16_color_twist_p3)
    }

    fn color_twist_with(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
        operation: PlanarColorTwist<u16>,
    ) -> Result<Self> {
        self.prepare_eager(stream_context)?;

        let mut destination =
            Self::create_uninit(stream_context.cuda_context(), self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe { operation(stream_context, &source, &mut destination_view, twist) }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}

impl PlanarImage<i8, 3> {
    /// 3 channel 8-bit signed planar color twist.
    ///
    /// Uses [`color_twist_i8_p3`](crate::image::raw::color::color_twist_i8_p3).
    /// Writes converted values into the supplied destination planes. Prepares both source and
    /// destination regions, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    /// Each source and destination image pair must have matching sizes.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    pub fn color_twist_into(
        stream_context: &StreamContext,
        source: &PlanarImageView<'_, i8, 3>,
        destination: &mut PlanarImageOutputView<'_, i8, 3>,
        twist: color::ColorTwistMatrix,
    ) -> Result<()> {
        stream_context.stream().ensure_not_capturing()?;
        for plane in source.planes() {
            stream_context.prepare_region(plane)?;
        }
        for plane in destination.planes() {
            stream_context.prepare_region(plane)?;
        }
        // SAFETY: the regions were prepared on this stream above, and `sync_scope` completes the work before the borrows end.
        stream_context.sync_scope(|| unsafe {
            color::color_twist_i8_p3(stream_context, source, destination, twist)
        })
    }
}

impl PlanarImage<i16, 3> {
    /// 3 channel 16-bit signed planar color twist.
    ///
    /// Uses [`color_twist_i16_p3`](crate::image::raw::color::color_twist_i16_p3).
    /// Allocates new owned full-resolution planes with the same dimensions as the source.
    /// Prepares every source region, waiting for producers on other streams.
    /// Rejects stream capture and returns after successful stream completion. Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding fails, a participating image belongs to another CUDA context, or NPP rejects the operation parameters.
    /// Each source and destination image pair must have matching sizes.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    /// Creating output allocations and constructing image or plane views can also fail.
    pub fn color_twist(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
    ) -> Result<Self> {
        self.color_twist_with(stream_context, twist, color::color_twist_i16_p3)
    }

    fn color_twist_with(
        self,
        stream_context: &StreamContext,
        twist: color::ColorTwistMatrix,
        operation: PlanarColorTwist<i16>,
    ) -> Result<Self> {
        self.prepare_eager(stream_context)?;

        let mut destination =
            Self::create_uninit(stream_context.cuda_context(), self.planes[0].size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output()?;
            stream_context.sync_scope(|| {
                // SAFETY: this private helper selects a matching native color producer and retains all participants.
                unsafe { operation(stream_context, &source, &mut destination_view, twist) }
            })?;
        }

        // SAFETY: completed color conversion established all logical output channels.
        Ok(unsafe { destination.assume_init() })
    }
}
