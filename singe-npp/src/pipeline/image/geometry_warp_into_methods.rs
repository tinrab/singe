use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, ImageViewMut},
};

use super::super::super::{
    ImagePipeline, WarpAffine, WarpPerspective, WarpQuad,
    geometry_dispatch::{
        WarpAffineBackImage, WarpAffineImage, WarpAffineQuadImage, WarpPerspectiveBackImage,
        WarpPerspectiveImage, WarpPerspectiveQuadImage,
    },
};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn warp_affine_into(
        stream_context: &StreamContext,
        warp: &WarpAffine,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>
    where
        Self: WarpAffineImage<T, L>,
    {
        <Self as WarpAffineImage<T, L>>::warp_affine_image(
            stream_context,
            warp,
            source,
            destination,
        )
    }

    pub fn warp_affine_back_into(
        stream_context: &StreamContext,
        warp: &WarpAffine,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>
    where
        Self: WarpAffineBackImage<T, L>,
    {
        <Self as WarpAffineBackImage<T, L>>::warp_affine_back_image(
            stream_context,
            warp,
            source,
            destination,
        )
    }

    pub fn warp_affine_quad_into(
        stream_context: &StreamContext,
        warp: &WarpQuad,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>
    where
        Self: WarpAffineQuadImage<T, L>,
    {
        <Self as WarpAffineQuadImage<T, L>>::warp_affine_quad_image(
            stream_context,
            warp,
            source,
            destination,
        )
    }

    pub fn warp_perspective_into(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>
    where
        Self: WarpPerspectiveImage<T, L>,
    {
        <Self as WarpPerspectiveImage<T, L>>::warp_perspective_image(
            stream_context,
            warp,
            source,
            destination,
        )
    }

    pub fn warp_perspective_back_into(
        stream_context: &StreamContext,
        warp: &WarpPerspective,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>
    where
        Self: WarpPerspectiveBackImage<T, L>,
    {
        <Self as WarpPerspectiveBackImage<T, L>>::warp_perspective_back_image(
            stream_context,
            warp,
            source,
            destination,
        )
    }

    pub fn warp_perspective_quad_into(
        stream_context: &StreamContext,
        warp: &WarpQuad,
        source: &ImageView<'_, T, L>,
        destination: &mut ImageViewMut<'_, T, L>,
    ) -> Result<()>
    where
        Self: WarpPerspectiveQuadImage<T, L>,
    {
        <Self as WarpPerspectiveQuadImage<T, L>>::warp_perspective_quad_image(
            stream_context,
            warp,
            source,
            destination,
        )
    }
}
