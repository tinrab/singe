use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageView, ImageViewMut},
};

use super::super::{
    Remap, ResizeSqrPixel, ResizeSqrPixelAdvanced, Rotate, WarpAffine, WarpPerspective, WarpQuad,
};

pub trait RotateImage<T, L> {
    fn rotate_image(
        stream_context: &StreamContext,
        rotate: &Rotate,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait ResizeSqrPixelImage<T, L> {
    fn resize_sqr_pixel_image(
        stream_context: &StreamContext,
        resize: &ResizeSqrPixel,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait ResizeSqrPixelAdvancedImage<T, L> {
    fn resize_sqr_pixel_advanced_image(
        stream_context: &StreamContext,
        resize: &ResizeSqrPixelAdvanced,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait RemapImage<T, M, L> {
    fn remap_image(
        stream_context: &StreamContext,
        remap: &Remap,
        source: &ImageView<'_, T, L>,
        x_map: &ImageView<'_, M, C1>,
        y_map: &ImageView<'_, M, C1>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait WarpAffineImage<T, L> {
    fn warp_affine_image(
        stream_context: &StreamContext,
        warp: &WarpAffine,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait WarpAffineBackImage<T, L> {
    fn warp_affine_back_image(
        stream_context: &StreamContext,
        warp: &WarpAffine,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait WarpAffineQuadImage<T, L> {
    fn warp_affine_quad_image(
        stream_context: &StreamContext,
        warp: &WarpQuad,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait WarpPerspectiveImage<T, L> {
    fn warp_perspective_image(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait WarpPerspectiveBackImage<T, L> {
    fn warp_perspective_back_image(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}

pub trait WarpPerspectiveQuadImage<T, L> {
    fn warp_perspective_quad_image(
        stream_context: &StreamContext,
        warp: &WarpQuad,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>;
}
