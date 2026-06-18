use crate::{
    context::StreamContext,
    error::Result,
    image::{color, view::C1},
    pipeline::{ImageAllocator, Workspace},
};

use super::super::planar::{SemiplanarImage, SubsampledPlanarImage};

impl SemiplanarImage<u8> {
    pub fn ycbcr411_to_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        Workspace: ImageAllocator<u8, C1>,
    {
        self.color_convert_to_subsampled_planar(stream_context, 4, 1, color::ycbcr411_u8_p2_to_p3)
    }

    pub fn ycbcr411_to_ycbcr422_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        Workspace: ImageAllocator<u8, C1>,
    {
        self.color_convert_to_subsampled_planar(
            stream_context,
            2,
            1,
            color::ycbcr411_to_ycbcr422_u8_p2_to_p3,
        )
    }

    pub fn ycbcr411_to_ycbcr420_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        Workspace: ImageAllocator<u8, C1>,
    {
        self.color_convert_to_subsampled_planar(
            stream_context,
            2,
            2,
            color::ycbcr411_to_ycbcr420_u8_p2_to_p3,
        )
    }

    pub fn ycbcr411_to_ycrcb420_planar(
        self,
        stream_context: &StreamContext,
    ) -> Result<SubsampledPlanarImage<u8>>
    where
        Workspace: ImageAllocator<u8, C1>,
    {
        self.color_convert_to_subsampled_planar(
            stream_context,
            2,
            2,
            color::ycbcr411_to_ycrcb420_u8_p2_to_p3,
        )
    }
}
