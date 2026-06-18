use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, C3, ImageView},
    },
    types::{BorderType, HistogramOfGradientsConfig, Point, Size},
};

use super::super::{ImagePipeline, filtering_traits::HistogramOfGradientsBorderImage};

macro_rules! impl_histogram_of_gradients_border_image {
    ($ty:ty, $layout:ty, $histogram:path) => {
        impl<'a> HistogramOfGradientsBorderImage<$ty, $layout> for ImagePipeline<'a, $ty, $layout> {
            fn histogram_of_gradients_border_image(
                stream_context: &StreamContext,
                source: &ImageView<'_, $ty, $layout>,
                source_offset: Point,
                locations: &[Point],
                descriptors: &mut DeviceMemory<f32>,
                roi: Size,
                config: HistogramOfGradientsConfig,
                scratch: &mut DeviceMemory<u8>,
                border_type: BorderType,
            ) -> Result<()> {
                $histogram(
                    stream_context,
                    source,
                    source_offset,
                    locations,
                    descriptors,
                    roi,
                    config,
                    scratch,
                    border_type,
                )
            }
        }
    };
}

impl_histogram_of_gradients_border_image!(
    u8,
    C1,
    filtering::histogram_of_gradients_border_u8_to_f32_c1
);
impl_histogram_of_gradients_border_image!(
    u8,
    C3,
    filtering::histogram_of_gradients_border_u8_to_f32_c3
);
impl_histogram_of_gradients_border_image!(
    u16,
    C1,
    filtering::histogram_of_gradients_border_u16_to_f32_c1
);
impl_histogram_of_gradients_border_image!(
    u16,
    C3,
    filtering::histogram_of_gradients_border_u16_to_f32_c3
);
impl_histogram_of_gradients_border_image!(
    i16,
    C1,
    filtering::histogram_of_gradients_border_i16_to_f32_c1
);
impl_histogram_of_gradients_border_image!(
    i16,
    C3,
    filtering::histogram_of_gradients_border_i16_to_f32_c3
);
impl_histogram_of_gradients_border_image!(f32, C1, filtering::histogram_of_gradients_border_f32_c1);
impl_histogram_of_gradients_border_image!(f32, C3, filtering::histogram_of_gradients_border_f32_c3);
