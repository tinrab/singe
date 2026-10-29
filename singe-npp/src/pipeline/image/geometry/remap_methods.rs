use std::sync::Arc;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image,
        view::{C1, ChannelLayout, ImageOutputView, ImageView},
    },
    pipeline::image::{ImageBacking, ImagePipeline, Remap, geometry_dispatch::RemapImage},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Samples the source at coordinates from `x_map` and `y_map`.
    ///
    /// `remap` selects the source ROI and interpolation. Each C1 coordinate map must have
    /// the destination image size. Map coordinates are passed to NPP without inspecting
    /// their values.
    /// Pixels outside the transformed source intersection, ignored alpha, and padding can
    /// remain unwritten. Initialize uncovered locations before reading a complete image.
    /// Prepares all participating image and coordinate-map regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source ROI must have positive dimensions and representable endpoints. Both
    /// coordinate maps must match the destination dimensions. NPP may reject geometry
    /// or interpolation parameters.
    pub fn remap_into<M>(
        stream_context: &StreamContext,
        remap: &Remap,
        source: &ImageView<'_, T, L>,
        x_map: &ImageView<'_, M, C1>,
        y_map: &ImageView<'_, M, C1>,
        destination: &mut ImageOutputView<'_, T, L>,
    ) -> Result<()>
    where
        M: Copy,
        Self: RemapImage<T, M, L>,
    {
        <Self as RemapImage<T, M, L>>::remap_image(
            stream_context,
            remap,
            source,
            x_map,
            y_map,
            destination,
        )
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Samples the source at coordinates from `x_map` and `y_map`.
    ///
    /// `remap` selects the source ROI and interpolation. Each C1 coordinate map must have
    /// the destination image size. Map coordinates are passed to NPP without inspecting
    /// their values.
    /// Takes ownership of the supplied initialized `destination` and returns it in the
    /// resulting pipeline. Pixels outside native writes and ignored alpha retain their
    /// destination values.
    /// Prepares all participating image and coordinate-map regions, waits for their producers
    /// on other streams, rejects capture, and returns after successful stream completion.
    /// Row padding remains unwritten.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// The source ROI must have positive dimensions and representable endpoints. Both
    /// coordinate maps must match the destination dimensions. NPP may reject geometry
    /// or interpolation parameters.
    /// Image view construction or capture-time destination retention can also fail.
    pub fn remap<M>(
        self,
        remap: Remap,
        x_map: &ImageView<'_, M, C1>,
        y_map: &ImageView<'_, M, C1>,
        mut destination: Image<T, L>,
    ) -> Result<Self>
    where
        M: Copy,
        Self: RemapImage<T, M, L>,
    {
        self.stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(destination.allocation()))?;
        self.ensure_eager()?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output_view()?;
            <Self as RemapImage<T, M, L>>::remap_image(
                self.stream_context,
                &remap,
                &source,
                x_map,
                y_map,
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
