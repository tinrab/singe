use std::sync::Arc;

use crate::{
    error::Result,
    image::{Image, geometry, view::C1},
    pipeline::image::geometry::{ImageBacking, ImagePipeline, ResizeSqrPixelAdvanced},
    workspace::ScratchBuffer,
};

impl ImagePipeline<'_, u8, C1> {
    /// Returns the device scratch capacity in bytes for advanced C1 `u8` resizing.
    ///
    /// Uses the source and destination ROI sizes and interpolation from `resize`. Queries
    /// NPP without allocating storage or submitting stream work.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the ROI sizes or interpolation, or converting the
    /// returned byte count to `usize` fails.
    pub fn resize_sqr_pixel_advanced_buffer_size(resize: &ResizeSqrPixelAdvanced) -> Result<usize> {
        geometry::resize_sqr_pixel_c1_advanced_buffer_size::<u8>(resize)
    }

    /// Resizes with explicit scale factors and advanced interpolation.
    ///
    /// `resize` supplies source and destination ROIs, independent horizontal and vertical
    /// scale factors, and interpolation. The provided implementation uses C1 `u8` images.
    /// Uses the supplied scratch buffer, checking its CUDA context and queried capacity.
    /// Takes ownership of the supplied initialized `destination` and returns it in the
    /// resulting pipeline. Pixels outside native writes and ignored alpha retain their
    /// destination values.
    /// Prepares all participating image regions and scratch, waits for their producers
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
    /// Scratch-size queries and checked size conversions can fail. Insufficient scratch
    /// capacity or a scratch context mismatch is rejected.
    /// Image view construction or capture-time destination retention can also fail.
    pub fn resize_sqr_pixel_advanced_with_scratch(
        self,
        resize: ResizeSqrPixelAdvanced,
        scratch: &mut ScratchBuffer,
        mut destination: Image<u8, C1>,
    ) -> Result<Self> {
        self.stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(destination.allocation()))?;
        self.ensure_eager()?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output_view()?;
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.prepare_region(&destination_view)?;
            self.stream_context.prepare_region(scratch)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: the initialized destination preserves untouched pixels; all native
                // participants remain borrowed through completion.
                unsafe {
                    geometry::resize_sqr_pixel_c1_advanced_with_scratch(
                        self.stream_context,
                        &resize,
                        &source,
                        &mut destination_view,
                        scratch,
                    )
                }
            })?;
        }

        Ok(Self {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
