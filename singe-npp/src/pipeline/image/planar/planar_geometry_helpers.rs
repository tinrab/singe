use std::sync::Arc;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry,
        view::{C1, ImageView, PlanarImageOutputView, PlanarImageView},
    },
    pipeline::image::planar::{PlanarImage, PlanarRemap},
};

impl<T, const C: usize> PlanarImage<T, C> {
    pub(crate) fn transform_geometry<G>(
        self,
        stream_context: &StreamContext,
        mut destination: Self,
        geometry: &G,
        operation: for<'source, 'destination> unsafe fn(
            &StreamContext,
            &G,
            &PlanarImageView<'source, T, C>,
            &mut PlanarImageOutputView<'destination, T, C>,
        ) -> Result<()>,
    ) -> Result<Self>
    where
        T: Copy,
    {
        for plane in destination.planes() {
            stream_context
                .stream()
                .defer_drop_until_capture_end(Arc::clone(plane.allocation()))?;
        }
        self.prepare_eager(stream_context)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output_view()?;
            stream_context.stream().ensure_not_capturing()?;
            for plane in source.planes() {
                stream_context.prepare_region(plane)?;
            }
            for plane in destination_view.planes() {
                stream_context.prepare_region(plane)?;
            }
            stream_context.sync_scope(|| {
                // SAFETY: checked native geometry dispatch retains these plane borrows through completion.
                unsafe { operation(stream_context, geometry, &source, &mut destination_view) }
            })?;
        }

        Ok(destination)
    }

    pub(crate) fn remap_geometry<M>(
        self,
        stream_context: &StreamContext,
        remap: &geometry::Remap,
        x_map: &ImageView<'_, M, C1>,
        y_map: &ImageView<'_, M, C1>,
        mut destination: Self,
        operation: PlanarRemap<T, M, C>,
    ) -> Result<Self>
    where
        T: Copy,
        M: Copy,
    {
        for plane in destination.planes() {
            stream_context
                .stream()
                .defer_drop_until_capture_end(Arc::clone(plane.allocation()))?;
        }
        self.prepare_eager(stream_context)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output_view()?;
            stream_context.stream().ensure_not_capturing()?;
            for plane in source.planes() {
                stream_context.prepare_region(plane)?;
            }
            for plane in destination_view.planes() {
                stream_context.prepare_region(plane)?;
            }
            stream_context.prepare_region(x_map)?;
            stream_context.prepare_region(y_map)?;
            stream_context.sync_scope(|| {
                // SAFETY: checked native geometry dispatch retains these plane and map borrows through completion.
                unsafe {
                    operation(
                        stream_context,
                        remap,
                        &source,
                        x_map,
                        y_map,
                        &mut destination_view,
                    )
                }
            })?;
        }

        Ok(destination)
    }
}
