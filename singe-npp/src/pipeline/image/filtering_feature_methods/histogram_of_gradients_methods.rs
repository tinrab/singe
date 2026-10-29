use std::mem::size_of;

use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::{
    error::{Error, Result},
    image::{filtering, view::ChannelLayout},
    pipeline::image::{
        HistogramOfGradients, ImagePipeline,
        filtering_feature_methods::HistogramOfGradientsBorderImage,
    },
    types::{BorderType, HistogramOfGradientsConfig, Point, Size},
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Extracts histogram-of-oriented-gradients descriptors into owned device memory.
    ///
    /// `locations` is a host slice of window starting locations; `roi` is the source
    /// processing extent beginning at `source_offset`. `config` determines cells, blocks,
    /// bins, and descriptor windows. Returns consecutively packed `f32` descriptor windows
    /// in row order and their total byte count. Allocates descriptor and scratch storage
    /// using the native size queries for the same request.
    /// Uses centered derivatives, unsigned orientations from 0 to 180 degrees, interpolation
    /// between neighboring bins, and overlapping blocks with L2 normalization. Performs no
    /// threshold clipping or Gaussian smoothing. For C3 input, the channel with the greatest
    /// gradient magnitude supplies each pixel gradient. NPP supports only replicate borders.
    /// Prepares participating regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail. The source offset
    /// must address a source pixel; output-sized ROI endpoints must not overflow or exceed
    /// the source image.
    /// Descriptor or scratch-size queries, allocation, or source view construction can fail.
    /// The location count must fit `i32`, ROI dimensions must be positive, and the descriptor
    /// byte count must be divisible by `size_of::<f32>()`. NPP can reject configurations,
    /// window locations, or border selections.
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
        self.ensure_eager()?;
        let descriptor_bytes =
            filtering::histogram_of_gradients_border_descriptors_size(config, locations.len())?;
        if !descriptor_bytes.is_multiple_of(size_of::<f32>()) {
            return Err(Error::out_of_range("hog descriptor byte size"));
        }
        let descriptor_count = descriptor_bytes / size_of::<f32>();
        let mut descriptors = DeviceMemory::<f32>::create_uninit(
            self.stream_context.cuda_context(),
            descriptor_count,
        )?;
        let scratch_bytes =
            filtering::histogram_of_gradients_border_buffer_size(config, locations, roi)?;
        let mut scratch =
            DeviceMemory::<u8>::create_uninit(self.stream_context.cuda_context(), scratch_bytes)?;

        {
            let source = self.view()?;
            <Self as HistogramOfGradientsBorderImage<T, L>>::histogram_of_gradients_border_image(
                self.stream_context,
                &source,
                source_offset,
                locations,
                &mut DeviceOutputView::from_output(&mut descriptors),
                roi,
                config,
                &mut DeviceOutputView::from_output(&mut scratch),
                border_type,
            )?;
        }

        Ok(HistogramOfGradients {
            // SAFETY: completed native dispatch writes exactly the queried contiguous float descriptors.
            descriptors: unsafe { descriptors.assume_init() },
            descriptor_bytes,
        })
    }
}
