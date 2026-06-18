use crate::{
    context::StreamContext,
    error::Result,
    image::{
        geometry,
        view::{C1, ImageView, PlanarImageView, PlanarImageViewMut},
    },
    pipeline::{ImageAllocator, Workspace},
    types::Size,
};

use super::{CreatePlanarImage, PlanarImage, PlanarRemap};

impl<T, const C: usize> PlanarImage<T, C> {
    pub(in crate::pipeline::image) fn transform_geometry<G>(
        self,
        stream_context: &StreamContext,
        destination_size: Size,
        geometry: &G,
        operation: for<'source, 'destination> fn(
            &StreamContext,
            &G,
            &PlanarImageView<'source, T, C>,
            &mut PlanarImageViewMut<'destination, T, C>,
        ) -> Result<()>,
    ) -> Result<Self>
    where
        T: Copy,
        Workspace: ImageAllocator<T, C1>,
        Self: CreatePlanarImage<T, C>,
    {
        let mut destination = <Self as CreatePlanarImage<T, C>>::create(destination_size)?;

        {
            let source = <Self as CreatePlanarImage<T, C>>::view(&self)?;
            let mut destination_view =
                <Self as CreatePlanarImage<T, C>>::view_mut(&mut destination)?;
            operation(stream_context, geometry, &source, &mut destination_view)?;
        }

        Ok(destination)
    }

    pub(in crate::pipeline::image) fn remap_geometry<M>(
        self,
        stream_context: &StreamContext,
        remap: &geometry::Remap,
        x_map: &ImageView<'_, M, C1>,
        y_map: &ImageView<'_, M, C1>,
        operation: PlanarRemap<T, M, C>,
    ) -> Result<Self>
    where
        T: Copy,
        M: Copy,
        Workspace: ImageAllocator<T, C1>,
        Self: CreatePlanarImage<T, C>,
    {
        let mut destination = <Self as CreatePlanarImage<T, C>>::create(x_map.size())?;

        {
            let source = <Self as CreatePlanarImage<T, C>>::view(&self)?;
            let mut destination_view =
                <Self as CreatePlanarImage<T, C>>::view_mut(&mut destination)?;
            operation(
                stream_context,
                remap,
                &source,
                x_map,
                y_map,
                &mut destination_view,
            )?;
        }

        Ok(destination)
    }
}
