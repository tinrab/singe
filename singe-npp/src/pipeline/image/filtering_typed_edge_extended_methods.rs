use crate::{
    error::Result,
    image::view::ChannelLayout,
    pipeline::{ImageAllocator, Workspace},
    types::MaskSize,
};

use super::super::{ImagePipeline, filtering::TypedSobelExtendedFilterImage};

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn filter_sobel_horizontal_second_to<D, M>(
        self,
        mask_size: MaskSize,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy,
        M: ChannelLayout,
        Workspace: ImageAllocator<D, M>,
        Self: TypedSobelExtendedFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_filter(
            mask_size,
            <Self as TypedSobelExtendedFilterImage<
                T,
                L,
                D,
                M,
            >>::filter_sobel_horizontal_second_to_image,
        )
    }

    pub fn filter_sobel_vertical_second_to<D, M>(
        self,
        mask_size: MaskSize,
    ) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy,
        M: ChannelLayout,
        Workspace: ImageAllocator<D, M>,
        Self: TypedSobelExtendedFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_filter(
            mask_size,
            <Self as TypedSobelExtendedFilterImage<
                T,
                L,
                D,
                M,
            >>::filter_sobel_vertical_second_to_image,
        )
    }

    pub fn filter_sobel_cross_to<D, M>(self, mask_size: MaskSize) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy,
        M: ChannelLayout,
        Workspace: ImageAllocator<D, M>,
        Self: TypedSobelExtendedFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_filter(
            mask_size,
            <Self as TypedSobelExtendedFilterImage<T, L, D, M>>::filter_sobel_cross_to_image,
        )
    }

    pub fn filter_laplace_to<D, M>(self, mask_size: MaskSize) -> Result<ImagePipeline<'a, D, M>>
    where
        D: Copy,
        M: ChannelLayout,
        Workspace: ImageAllocator<D, M>,
        Self: TypedSobelExtendedFilterImage<T, L, D, M>,
    {
        self.typed_masked_edge_directional_filter(
            mask_size,
            <Self as TypedSobelExtendedFilterImage<T, L, D, M>>::filter_laplace_to_image,
        )
    }
}
