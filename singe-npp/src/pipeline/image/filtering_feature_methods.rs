use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, ImageNormalization, MaskSize, Point},
};

use super::{GradientVector, ImagePipeline, filtering::*};

#[path = "filtering_histogram_of_gradients_methods.rs"]
mod histogram_of_gradients_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn gradient_vector_prewitt_border_to<G>(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<GradientVector<G>>
    where
        G: Copy,
        Workspace: ImageAllocator<G, C1> + ImageAllocator<f32, C1>,
        Self: GradientVectorBorderImage<T, L, G>,
    {
        self.gradient_vector_border(
            source_offset,
            mask_size,
            norm,
            border_type,
            <Self as GradientVectorBorderImage<T, L, G>>::gradient_vector_prewitt_border_image,
        )
    }

    pub fn gradient_vector_scharr_border_to<G>(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<GradientVector<G>>
    where
        G: Copy,
        Workspace: ImageAllocator<G, C1> + ImageAllocator<f32, C1>,
        Self: GradientVectorBorderImage<T, L, G>,
    {
        self.gradient_vector_border(
            source_offset,
            mask_size,
            norm,
            border_type,
            <Self as GradientVectorBorderImage<T, L, G>>::gradient_vector_scharr_border_image,
        )
    }

    pub fn gradient_vector_sobel_border_to<G>(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
    ) -> Result<GradientVector<G>>
    where
        G: Copy,
        Workspace: ImageAllocator<G, C1> + ImageAllocator<f32, C1>,
        Self: GradientVectorBorderImage<T, L, G>,
    {
        self.gradient_vector_border(
            source_offset,
            mask_size,
            norm,
            border_type,
            <Self as GradientVectorBorderImage<T, L, G>>::gradient_vector_sobel_border_image,
        )
    }

    fn gradient_vector_border<G>(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        norm: ImageNormalization,
        border_type: BorderType,
        operation: fn(
            &StreamContext,
            &ImageView<'_, T, L>,
            Point,
            &mut ImageViewMut<'_, G, C1>,
            &mut ImageViewMut<'_, G, C1>,
            &mut ImageViewMut<'_, G, C1>,
            &mut ImageViewMut<'_, f32, C1>,
            MaskSize,
            ImageNormalization,
            BorderType,
        ) -> Result<()>,
    ) -> Result<GradientVector<G>>
    where
        G: Copy,
        Workspace: ImageAllocator<G, C1> + ImageAllocator<f32, C1>,
    {
        let size = self.size();
        let mut x = self.workspace.image::<G, C1>(size)?;
        let mut y = self.workspace.image::<G, C1>(size)?;
        let mut magnitude = self.workspace.image::<G, C1>(size)?;
        let mut angle = self.workspace.image::<f32, C1>(size)?;

        {
            let source = self.view()?;
            let mut x_view = x.view_mut()?;
            let mut y_view = y.view_mut()?;
            let mut magnitude_view = magnitude.view_mut()?;
            let mut angle_view = angle.view_mut()?;
            operation(
                self.stream_context,
                &source,
                source_offset,
                &mut x_view,
                &mut y_view,
                &mut magnitude_view,
                &mut angle_view,
                mask_size,
                norm,
                border_type,
            )?;
        }

        Ok(GradientVector {
            x,
            y,
            magnitude,
            angle,
        })
    }
}
