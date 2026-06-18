use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{
    ImageBacking, ImagePipeline, filtering_distance_traits::SignedDistanceTransformPbaImage,
};

#[path = "filtering_signed_distance_antialiasing_methods.rs"]
mod antialiasing_methods;
#[path = "filtering_signed_distance_into_methods.rs"]
mod into_methods;

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
{
    pub fn signed_distance_transform_pba<D>(
        self,
        cutoff_value: T,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
    ) -> Result<ImagePipeline<'a, D, C1>>
    where
        D: Copy,
        Workspace: ImageAllocator<D, C1>,
        Self: SignedDistanceTransformPbaImage<T, D>,
    {
        self.signed_distance_transform_pba_operation(
            cutoff_value,
            subpixel_x_shift,
            subpixel_y_shift,
            <Self as SignedDistanceTransformPbaImage<T, D>>::signed_distance_transform_pba_image,
        )
    }

    pub fn signed_distance_transform_abs_pba<D>(
        self,
        cutoff_value: T,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
    ) -> Result<ImagePipeline<'a, D, C1>>
    where
        D: Copy,
        Workspace: ImageAllocator<D, C1>,
        Self: SignedDistanceTransformPbaImage<T, D>,
    {
        self.signed_distance_transform_pba_operation(
            cutoff_value,
            subpixel_x_shift,
            subpixel_y_shift,
            <Self as SignedDistanceTransformPbaImage<T, D>>::signed_distance_transform_abs_pba_image,
        )
    }

    fn signed_distance_transform_pba_operation<D>(
        self,
        cutoff_value: T,
        subpixel_x_shift: f64,
        subpixel_y_shift: f64,
        operation: fn(
            &StreamContext,
            &ImageView<'_, T, C1>,
            T,
            f64,
            f64,
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
                cutoff_value,
                subpixel_x_shift,
                subpixel_y_shift,
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
