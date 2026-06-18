use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{
    ImagePipeline, WarpAffine, WarpPerspective, WarpQuad,
    geometry_dispatch::{
        WarpAffineBackImage, WarpAffineImage, WarpAffineQuadImage, WarpPerspectiveBackImage,
        WarpPerspectiveImage, WarpPerspectiveQuadImage,
    },
};

#[path = "geometry_warp_into_methods.rs"]
mod into_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: WarpAffineImage<T, L>,
{
    pub fn warp_affine(self, warp: WarpAffine) -> Result<Self> {
        self.transform_geometry(
            warp.destination_roi.size(),
            &warp,
            <Self as WarpAffineImage<T, L>>::warp_affine_image,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: WarpAffineBackImage<T, L>,
{
    pub fn warp_affine_back(self, warp: WarpAffine) -> Result<Self> {
        self.transform_geometry(
            warp.destination_roi.size(),
            &warp,
            <Self as WarpAffineBackImage<T, L>>::warp_affine_back_image,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: WarpAffineQuadImage<T, L>,
{
    pub fn warp_affine_quad(self, warp: WarpQuad) -> Result<Self> {
        self.transform_geometry(
            warp.destination_roi.size(),
            &warp,
            <Self as WarpAffineQuadImage<T, L>>::warp_affine_quad_image,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: WarpPerspectiveImage<T, L>,
{
    pub fn warp_perspective(self, warp: WarpPerspective) -> Result<Self> {
        self.transform_geometry(
            warp.destination_roi.size(),
            &warp,
            <Self as WarpPerspectiveImage<T, L>>::warp_perspective_image,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: WarpPerspectiveBackImage<T, L>,
{
    pub fn warp_perspective_back(self, warp: WarpPerspective) -> Result<Self> {
        self.transform_geometry(
            warp.destination_roi.size(),
            &warp,
            <Self as WarpPerspectiveBackImage<T, L>>::warp_perspective_back_image,
        )
    }
}

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: WarpPerspectiveQuadImage<T, L>,
{
    pub fn warp_perspective_quad(self, warp: WarpQuad) -> Result<Self> {
        self.transform_geometry(
            warp.destination_roi.size(),
            &warp,
            <Self as WarpPerspectiveQuadImage<T, L>>::warp_perspective_quad_image,
        )
    }
}
