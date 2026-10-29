use singe_cuda::view::DeviceOutput;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, ImageView},
    pipeline::image::{
        ImagePipeline,
        statistics::{
            HistogramEvenChannelsImage, HistogramRangeChannelsImage, ImageHistograms,
            ImageStatistic,
        },
    },
};

mod helpers;

pub use helpers::{histogram_buffers, histogram_range_buffers, image_histograms};

mod c3_range_channel_methods;
mod c4_channel_methods;

impl<T> ImagePipeline<'_, T, C3>
where
    T: Copy,
    Self: HistogramEvenChannelsImage<T, C3, 3>,
{
    /// Computes per-channel histograms using evenly spaced bins.
    /// Processes all three C3 channels. Each channel has its own
    /// level count, range bounds, and histogram.
    /// The lower bound is inclusive and the upper bound is exclusive. Each level count
    /// specifies one more boundary than bins, yielding `levels - 1` bins per channel. The
    /// wrapper forwards bounds to NPP without validating their order.
    /// Each supplied `i32` histogram output must have exactly its channel bin count.
    /// Prepares source and result regions and waits for producers on other streams. Rejects
    /// capture, queries and allocates temporary device scratch, and returns after successful
    /// stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or source/result region validation,
    /// producer waits, scratch-size queries or allocation, native execution, or stream
    /// synchronization fail.
    /// Each level count must be at least two.
    /// Each histogram output must contain exactly its level count minus one entries.
    pub fn histogram_even_channels_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        histograms: [&mut (impl DeviceOutput<i32> + ?Sized); 3],
        levels: [i32; 3],
        lower_levels: [i32; 3],
        upper_levels: [i32; 3],
    ) -> Result<()> {
        <Self as HistogramEvenChannelsImage<T, C3, 3>>::histogram_even_channels(
            stream_context,
            source,
            histograms,
            levels,
            lower_levels,
            upper_levels,
        )
    }

    /// Computes per-channel histograms using evenly spaced bins.
    /// Processes all three C3 channels. Each channel has its own
    /// level count, range bounds, and histogram.
    /// The lower bound is inclusive and the upper bound is exclusive. Each level count
    /// specifies one more boundary than bins, yielding `levels - 1` bins per channel. The
    /// wrapper forwards bounds to NPP without validating their order.
    /// Allocates 3 owned device histograms containing the corresponding `i32` bin counts.
    /// Prepares source and result regions and waits for producers on other streams. Rejects
    /// capture, queries and allocates temporary device scratch, and returns after successful
    /// stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or source/result region validation,
    /// producer waits, scratch-size queries or allocation, native execution, or stream
    /// synchronization fail.
    /// Each level count must be at least two.
    /// Result allocation or source view construction can also fail.
    pub fn histogram_even_channels(
        self,
        levels: [i32; 3],
        lower_levels: [i32; 3],
        upper_levels: [i32; 3],
    ) -> Result<ImageHistograms> {
        let [mut h0, mut h1, mut h2] =
            histogram_buffers(self.stream_context.cuda_context(), levels)?;
        {
            let source = self.view()?;
            <Self as HistogramEvenChannelsImage<T, C3, 3>>::histogram_even_channels(
                self.stream_context,
                &source,
                [&mut h0, &mut h1, &mut h2],
                levels,
                lower_levels,
                upper_levels,
            )?;
        }
        // SAFETY: the histogram calls above queued a producer for every bin on this stream.
        unsafe { image_histograms(self.stream_context, vec![h0, h1, h2]) }
    }
}
