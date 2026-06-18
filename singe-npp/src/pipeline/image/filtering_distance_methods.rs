use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::filtering_traits::DistanceTransformPbaImage;
use super::{ImageBacking, ImagePipeline};

#[path = "filtering_distance_antialiasing_into_methods.rs"]
mod antialiasing_into_methods;

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
{
    pub fn distance_transform_pba_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageViewMut<'_, D, C1>,
    ) -> Result<()>
    where
        D: Copy,
        Self: DistanceTransformPbaImage<T, D>,
    {
        <Self as DistanceTransformPbaImage<T, D>>::distance_transform_pba_image(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination,
        )
    }

    pub fn distance_transform_abs_pba_into<D>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        minimum_site_value: T,
        maximum_site_value: T,
        destination: &mut ImageViewMut<'_, D, C1>,
    ) -> Result<()>
    where
        D: Copy,
        Self: DistanceTransformPbaImage<T, D>,
    {
        <Self as DistanceTransformPbaImage<T, D>>::distance_transform_abs_pba_image(
            stream_context,
            source,
            minimum_site_value,
            maximum_site_value,
            destination,
        )
    }

    pub fn distance_transform_pba<D>(
        self,
        minimum_site_value: T,
        maximum_site_value: T,
    ) -> Result<ImagePipeline<'a, D, C1>>
    where
        D: Copy,
        Workspace: ImageAllocator<D, C1>,
        Self: DistanceTransformPbaImage<T, D>,
    {
        self.distance_transform_pba_operation(
            minimum_site_value,
            maximum_site_value,
            <Self as DistanceTransformPbaImage<T, D>>::distance_transform_pba_image,
        )
    }

    pub fn distance_transform_abs_pba<D>(
        self,
        minimum_site_value: T,
        maximum_site_value: T,
    ) -> Result<ImagePipeline<'a, D, C1>>
    where
        D: Copy,
        Workspace: ImageAllocator<D, C1>,
        Self: DistanceTransformPbaImage<T, D>,
    {
        self.distance_transform_pba_operation(
            minimum_site_value,
            maximum_site_value,
            <Self as DistanceTransformPbaImage<T, D>>::distance_transform_abs_pba_image,
        )
    }

    fn distance_transform_pba_operation<D>(
        self,
        minimum_site_value: T,
        maximum_site_value: T,
        operation: fn(
            &StreamContext,
            &ImageView<'_, T, C1>,
            T,
            T,
            &mut ImageViewMut<'_, D, C1>,
        ) -> Result<()>,
    ) -> Result<ImagePipeline<'a, D, C1>>
    where
        D: Copy,
        Workspace: ImageAllocator<D, C1>,
    {
        let mut destination = self.workspace.image::<D, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            operation(
                self.stream_context,
                &source,
                minimum_site_value,
                maximum_site_value,
                &mut destination_view,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
