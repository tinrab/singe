use std::sync::Arc;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image,
        view::{ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{
        ImageBacking, ImagePipeline, ResizeSqrPixel, ResizeSqrPixelAdvanced, Rotate,
    },
};

type GeometricTransformOperation<T, G, L> =
    fn(&StreamContext, &G, &ImageView<'_, T, L>, &mut ImageOutputView<'_, T, L>) -> Result<()>;

use crate::pipeline::image::geometry_dispatch::{
    ResizeSqrPixelAdvancedImage, ResizeSqrPixelImage, RotateImage,
};

mod remap_methods;
mod resize_advanced_scratch_methods;
mod warp_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    fn transform_geometry<G>(
        self,
        mut destination: Image<T, L>,
        geometry: &G,
        operation: GeometricTransformOperation<T, G, L>,
    ) -> Result<Self> {
        self.stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(destination.allocation()))?;
        self.ensure_eager()?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output_view()?;
            operation(
                self.stream_context,
                geometry,
                &source,
                &mut destination_view,
            )?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: RotateImage<T, L>,
{
    /// Rotates the source ROI using the configured angle, shifts, and interpolation.
    ///
    /// `rotate.angle` is in degrees and the shifts are in pixels. Source and destination
    /// ROIs select the native processing regions.
    /// Takes ownership of the supplied initialized `destination` and returns it in the
    /// resulting pipeline. Pixels outside native writes and ignored alpha retain their
    /// destination values.
    /// Prepares all participating image regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source ROI must have positive dimensions and representable endpoints. The
    /// destination ROI must fit within the destination image. NPP may reject geometry
    /// or interpolation parameters.
    /// Image view construction or capture-time destination retention can also fail.
    pub fn rotate(self, rotate: Rotate, destination: Image<T, L>) -> Result<Self> {
        self.transform_geometry(
            destination,
            &rotate,
            <Self as RotateImage<T, L>>::rotate_image,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Resizes with explicit square-pixel scale factors and shifts.
    ///
    /// `resize` supplies source and destination ROIs, independent horizontal and vertical
    /// scale factors, shifts in pixels, and interpolation.
    /// Pixels outside the transformed source intersection, ignored alpha, and padding can
    /// remain unwritten. Initialize uncovered locations before reading a complete image.
    /// Prepares all participating image regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source ROI must have positive dimensions and representable endpoints. The
    /// destination ROI must fit within the destination image. NPP may reject geometry
    /// or interpolation parameters.
    pub fn resize_sqr_pixel_into(
        stream_context: &StreamContext,
        resize: &ResizeSqrPixel,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>
    where
        Self: ResizeSqrPixelImage<T, L>,
    {
        <Self as ResizeSqrPixelImage<T, L>>::resize_sqr_pixel_image(
            stream_context,
            resize,
            source,
            destination,
        )
    }

    /// Resizes with explicit scale factors and advanced interpolation.
    ///
    /// `resize` supplies source and destination ROIs, independent horizontal and vertical
    /// scale factors, and interpolation. The provided implementation uses C1 `u8` images.
    /// Allocates temporary device scratch using the native size query.
    /// Pixels outside the transformed source intersection, ignored alpha, and padding can
    /// remain unwritten. Initialize uncovered locations before reading a complete image.
    /// Prepares all participating image regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source ROI must have positive dimensions and representable endpoints. The
    /// destination ROI must fit within the destination image. NPP may reject geometry
    /// or interpolation parameters.
    /// Scratch-size queries and checked size conversions can fail. Scratch allocation can fail.
    pub fn resize_sqr_pixel_advanced_into(
        stream_context: &StreamContext,
        resize: &ResizeSqrPixelAdvanced,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>
    where
        Self: ResizeSqrPixelAdvancedImage<T, L>,
    {
        <Self as ResizeSqrPixelAdvancedImage<T, L>>::resize_sqr_pixel_advanced_image(
            stream_context,
            resize,
            source,
            destination,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ResizeSqrPixelAdvancedImage<T, L>,
{
    /// Resizes with explicit scale factors and advanced interpolation.
    ///
    /// `resize` supplies source and destination ROIs, independent horizontal and vertical
    /// scale factors, and interpolation. The provided implementation uses C1 `u8` images.
    /// Allocates temporary device scratch using the native size query.
    /// Takes ownership of the supplied initialized `destination` and returns it in the
    /// resulting pipeline. Pixels outside native writes and ignored alpha retain their
    /// destination values.
    /// Prepares all participating image regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source ROI must have positive dimensions and representable endpoints. The
    /// destination ROI must fit within the destination image. NPP may reject geometry
    /// or interpolation parameters.
    /// Scratch-size queries and checked size conversions can fail. Scratch allocation can fail.
    /// Image view construction or capture-time destination retention can also fail.
    pub fn resize_sqr_pixel_advanced(
        self,
        resize: ResizeSqrPixelAdvanced,
        destination: Image<T, L>,
    ) -> Result<Self> {
        self.transform_geometry(
            destination,
            &resize,
            <Self as ResizeSqrPixelAdvancedImage<T, L>>::resize_sqr_pixel_advanced_image,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: ResizeSqrPixelImage<T, L>,
{
    /// Resizes with explicit square-pixel scale factors and shifts.
    ///
    /// `resize` supplies source and destination ROIs, independent horizontal and vertical
    /// scale factors, shifts in pixels, and interpolation.
    /// Takes ownership of the supplied initialized `destination` and returns it in the
    /// resulting pipeline. Pixels outside native writes and ignored alpha retain their
    /// destination values.
    /// Prepares all participating image regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source ROI must have positive dimensions and representable endpoints. The
    /// destination ROI must fit within the destination image. NPP may reject geometry
    /// or interpolation parameters.
    /// Image view construction or capture-time destination retention can also fail.
    pub fn resize_sqr_pixel(
        self,
        resize: ResizeSqrPixel,
        destination: Image<T, L>,
    ) -> Result<Self> {
        self.transform_geometry(
            destination,
            &resize,
            <Self as ResizeSqrPixelImage<T, L>>::resize_sqr_pixel_image,
        )
    }
}
