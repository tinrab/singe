use singe_cuda::types::f16;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        arithmetic,
        view::{C1, C3, ImageView, ImageViewMut},
    },
};

use super::super::{ImagePipeline, operation_traits::NaturalLogarithmImage};

macro_rules! impl_natural_logarithm_image {
    ($ty:ty, $layout:ty, $natural_logarithm:path, $natural_logarithm_in_place:path) => {
        impl<'a> NaturalLogarithmImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn natural_logarithm_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $natural_logarithm(stream_context, source, destination)
            }

            fn natural_logarithm_image_in_place(
                stream_context: &StreamContext,
                source_destination: &mut ImageViewMut<'_, $ty, $layout>,
            ) -> Result<()> {
                $natural_logarithm_in_place(stream_context, source_destination)
            }
        }
    };
}

impl_natural_logarithm_image!(
    f16,
    C1,
    arithmetic::natural_logarithm_f16_c1,
    arithmetic::natural_logarithm_f16_c1_in_place
);
impl_natural_logarithm_image!(
    f16,
    C3,
    arithmetic::natural_logarithm_f16_c3,
    arithmetic::natural_logarithm_f16_c3_in_place
);
