use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
    types::{BorderType, MaskSize, Point},
};

use super::super::super::{ImagePipeline, filtering::TypedSobelExtendedBorderFilterImage};

#[path = "filtering_typed_edge_laplace_border_methods.rs"]
mod laplace_methods;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn filter_sobel_horizontal_second_border_to<D, M>(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy,
        M: ChannelLayout,
        Workspace: ImageAllocator<D, M>,
        Self: TypedSobelExtendedBorderFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as TypedSobelExtendedBorderFilterImage<
                T,
                L,
                D,
                M,
            >>::filter_sobel_horizontal_second_border_to_image,
        )
    }

    pub fn filter_sobel_vertical_second_border_to<D, M>(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy,
        M: ChannelLayout,
        Workspace: ImageAllocator<D, M>,
        Self: TypedSobelExtendedBorderFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as TypedSobelExtendedBorderFilterImage<
                T,
                L,
                D,
                M,
            >>::filter_sobel_vertical_second_border_to_image,
        )
    }

    pub fn filter_sobel_cross_border_to<D, M>(
        self,
        source_offset: Point,
        mask_size: MaskSize,
        border_type: BorderType,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy,
        M: ChannelLayout,
        Workspace: ImageAllocator<D, M>,
        Self: TypedSobelExtendedBorderFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_border_filter(
            source_offset,
            mask_size,
            border_type,
            <Self as TypedSobelExtendedBorderFilterImage<
                T,
                L,
                D,
                M,
            >>::filter_sobel_cross_border_to_image,
        )
    }
}
