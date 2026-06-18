use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, MaskSize, Point},
};

use super::super::super::{ImagePipeline, filtering::SobelExtendedBorderFilterImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Workspace: ImageAllocator<T, L>,
    Self: SobelExtendedBorderFilterImage<T, L>,
{
    pub fn filter_sobel_horizontal_second_border(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.sobel_extended_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as SobelExtendedBorderFilterImage<
                T,
                L,
            >>::filter_sobel_horizontal_second_border_image,
        )
    }

    pub fn filter_sobel_vertical_second_border(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.sobel_extended_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as SobelExtendedBorderFilterImage<
                T,
                L,
            >>::filter_sobel_vertical_second_border_image,
        )
    }

    pub fn filter_sobel_cross_border(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<Self> {
        self.sobel_extended_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as SobelExtendedBorderFilterImage<T, L>>::filter_sobel_cross_border_image,
        )
    }
}
