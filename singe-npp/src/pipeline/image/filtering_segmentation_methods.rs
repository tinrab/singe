use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
    types::ImageNormalization,
};

#[path = "filtering_contours_methods.rs"]
mod contours_methods;
#[path = "filtering_distance_methods.rs"]
mod distance_methods;
#[path = "filtering_marker_compression_methods.rs"]
mod marker_compression_methods;

use super::{ImageBacking, ImagePipeline, filtering::*};

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
    Self: LabelMarkersUfImage<T>,
{
    pub fn label_markers_uf_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageViewMut<'_, u32, C1>,
        norm: ImageNormalization,
    ) -> Result<()> {
        <Self as LabelMarkersUfImage<T>>::label_markers_uf_image(
            stream_context,
            source,
            destination,
            norm,
        )
    }
}

impl<'a, T> ImagePipeline<'a, T, C1>
where
    T: Copy,
{
    pub fn label_markers_uf_batch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, C1>],
        destinations: &mut [ImageViewMut<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()>
    where
        Self: LabelMarkersUfBatchImage<T>,
    {
        <Self as LabelMarkersUfBatchImage<T>>::label_markers_uf_batch_image(
            stream_context,
            sources,
            destinations,
            norm,
        )
    }

    pub fn label_markers_uf_batch_advanced(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, C1>],
        destinations: &mut [ImageViewMut<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()>
    where
        Self: LabelMarkersUfBatchImage<T>,
    {
        <Self as LabelMarkersUfBatchImage<T>>::label_markers_uf_batch_advanced_image(
            stream_context,
            sources,
            destinations,
            norm,
        )
    }

    pub fn label_markers_uf(self, norm: ImageNormalization) -> Result<ImagePipeline<'a, u32, C1>>
    where
        Workspace: ImageAllocator<u32, C1>,
        Self: LabelMarkersUfImage<T>,
    {
        let mut destination = self.workspace.image::<u32, C1>(self.size())?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
            <Self as LabelMarkersUfImage<T>>::label_markers_uf_image(
                self.stream_context,
                &source,
                &mut destination_view,
                norm,
            )?;
        }

        Ok(ImagePipeline {
            stream_context: self.stream_context,
            workspace: self.workspace,
            backing: ImageBacking::Owned(destination),
        })
    }
}
