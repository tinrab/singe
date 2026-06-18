use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
    pipeline::{ImageAllocator, Workspace},
};

use super::{ImageBacking, ImagePipeline, ResizeSqrPixel, ResizeSqrPixelAdvanced, Rotate};

type GeometricTransformOperation<T, G, L> =
    fn(&StreamContext, &G, &ImageView<'_, T, L>, &mut ImageViewMut<'_, T, L>) -> Result<()>;

use super::geometry_dispatch::{ResizeSqrPixelAdvancedImage, ResizeSqrPixelImage, RotateImage};

#[path = "geometry_remap_methods.rs"]
mod remap_methods;
#[path = "geometry_resize_advanced_scratch_methods.rs"]
mod resize_advanced_scratch_methods;
#[path = "geometry_warp_methods.rs"]
mod warp_methods;
impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
{
    fn transform_geometry<G>(
        self,
        destination_size: crate::types::Size,
        geometry: &G,
        operation: GeometricTransformOperation<T, G, L>,
    ) -> Result<Self> {
        let mut destination = self.workspace.image::<T, L>(destination_size)?;

        {
            let source = self.view()?;
            let mut destination_view = destination.view_mut()?;
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

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: RotateImage<T, L>,
{
    pub fn rotate(self, rotate: Rotate) -> Result<Self> {
        self.transform_geometry(
            rotate.destination_roi.size(),
            &rotate,
            <Self as RotateImage<T, L>>::rotate_image,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn resize_sqr_pixel_into(
        stream_context: &StreamContext,
        resize: &ResizeSqrPixel,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
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

    pub fn resize_sqr_pixel_advanced_into(
        stream_context: &StreamContext,
        resize: &ResizeSqrPixelAdvanced,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
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

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: ResizeSqrPixelAdvancedImage<T, L>,
{
    pub fn resize_sqr_pixel_advanced(self, resize: ResizeSqrPixelAdvanced) -> Result<Self> {
        self.transform_geometry(
            resize.destination_roi.size(),
            &resize,
            <Self as ResizeSqrPixelAdvancedImage<T, L>>::resize_sqr_pixel_advanced_image,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: ResizeSqrPixelImage<T, L>,
{
    pub fn resize_sqr_pixel(self, resize: ResizeSqrPixel) -> Result<Self> {
        self.transform_geometry(
            resize.destination_roi.size(),
            &resize,
            <Self as ResizeSqrPixelImage<T, L>>::resize_sqr_pixel_image,
        )
    }
}
