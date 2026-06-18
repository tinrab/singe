use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
};

use super::super::{ImagePipeline, filtering::TypedEdgeDirectionalFilterImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn filter_scharr_horizontal_to<D, M>(self) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy,
        M: ChannelLayout,
        Workspace: ImageAllocator<D, M>,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        self.typed_edge_directional_filter(
            <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_scharr_horizontal_to_image,
        )
    }

    pub fn filter_scharr_vertical_to<D, M>(self) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy,
        M: ChannelLayout,
        Workspace: ImageAllocator<D, M>,
        Self: TypedEdgeDirectionalFilterImage<T, L, D, M>,
    {
        self.typed_edge_directional_filter(
            <Self as TypedEdgeDirectionalFilterImage<T, L, D, M>>::filter_scharr_vertical_to_image,
        )
    }
}
