use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    },
    types::MaskSize,
};

use super::{ImagePipeline, filtering_traits::*};

macro_rules! impl_high_low_gauss_filter_image {
    ($ty:ty, $layout:ty, $high_pass:path, $low_pass:path, $gauss:path) => {
        impl<'a> HighLowGaussFilterImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn filter_high_pass_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $high_pass(stream_context, source, destination, mask_size)
            }

            fn filter_low_pass_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $low_pass(stream_context, source, destination, mask_size)
            }

            fn filter_gauss_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
                mask_size: MaskSize,
            ) -> Result<()> {
                $gauss(stream_context, source, destination, mask_size)
            }
        }
    };
}

impl_high_low_gauss_filter_image!(
    u16,
    C1,
    filtering::filter_high_pass_u16_c1,
    filtering::filter_low_pass_u16_c1,
    filtering::filter_gauss_u16_c1
);
impl_high_low_gauss_filter_image!(
    u16,
    C3,
    filtering::filter_high_pass_u16_c3,
    filtering::filter_low_pass_u16_c3,
    filtering::filter_gauss_u16_c3
);
impl_high_low_gauss_filter_image!(
    u16,
    C4,
    filtering::filter_high_pass_u16_c4,
    filtering::filter_low_pass_u16_c4,
    filtering::filter_gauss_u16_c4
);
impl_high_low_gauss_filter_image!(
    u16,
    AC4,
    filtering::filter_high_pass_u16_ac4,
    filtering::filter_low_pass_u16_ac4,
    filtering::filter_gauss_u16_ac4
);
