use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, Point},
};

use super::super::{ImagePipeline, filtering::EdgeDirectionalBorderFilterImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: EdgeDirectionalBorderFilterImage<T, L>,
{
    pub fn filter_roberts_down_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.edge_directional_border_filter(
            source_offset,
            border_type,
            <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_roberts_down_border_image,
        )
    }

    pub fn filter_roberts_up_border(
        self,
        source_offset: Point,
        border_type: BorderType,
    ) -> Result<Self> {
        self.edge_directional_border_filter(
            source_offset,
            border_type,
            <Self as EdgeDirectionalBorderFilterImage<T, L>>::filter_roberts_up_border_image,
        )
    }
}
