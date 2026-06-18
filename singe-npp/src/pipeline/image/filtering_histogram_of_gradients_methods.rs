use std::mem::size_of;

use singe_cuda::memory::DeviceMemory;

use crate::{
    error::Result,
    image::{filtering, view::ChannelLayout},
    types::{BorderType, HistogramOfGradientsConfig, Point, Size},
};

use super::super::{HistogramOfGradients, ImagePipeline};
use super::HistogramOfGradientsBorderImage;

impl<'a, T, L> ImagePipeline<'a, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    pub fn histogram_of_gradients_border(
        self,
        source_offset: Point,
        locations: &[Point],
        roi: Size,
        config: HistogramOfGradientsConfig,
        border_type: BorderType,
    ) -> Result<HistogramOfGradients>
    where
        Self: HistogramOfGradientsBorderImage<T, L>,
    {
        let descriptor_bytes =
            filtering::histogram_of_gradients_border_descriptors_size(config, locations.len())?;
        let descriptor_count = descriptor_bytes.div_ceil(size_of::<f32>());
        let mut descriptors = DeviceMemory::<f32>::create(descriptor_count)?;
        let scratch_bytes =
            filtering::histogram_of_gradients_border_buffer_size(config, locations, roi)?;
        let mut scratch = DeviceMemory::<u8>::create(scratch_bytes)?;

        {
            let source = self.view()?;
            <Self as HistogramOfGradientsBorderImage<T, L>>::histogram_of_gradients_border_image(
                self.stream_context,
                &source,
                source_offset,
                locations,
                &mut descriptors,
                roi,
                config,
                &mut scratch,
                border_type,
            )?;
        }

        Ok(HistogramOfGradients {
            descriptors,
            descriptor_bytes,
        })
    }
}
