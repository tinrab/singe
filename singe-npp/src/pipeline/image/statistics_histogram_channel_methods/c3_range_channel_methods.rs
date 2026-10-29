use singe_cuda::view::DeviceOutput;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C3, ImageView},
    pipeline::image::statistics_histogram_channel_methods::{
        HistogramRangeChannelsImage, ImageHistograms, ImagePipeline, histogram_range_buffers,
        image_histograms,
    },
};

impl<T> ImagePipeline<'_, T, C3>
where
    T: Copy,
{
    /// Computes per-channel histograms using caller-supplied bin boundaries.
    /// Processes all three C3 channels. Each channel has its own
    /// boundary array and histogram.
    /// Bin `k` counts samples satisfying `levels[k] <= sample < levels[k + 1]` for that channel.
    /// Each histogram contains exactly one fewer bin than its boundary array. The provided
    /// integer-image implementations use `i32` boundaries; `f32` images use `f32` boundaries.
    /// Passes boundary slice addresses directly to NPP as device pointers without uploading
    /// them. Storage must remain GPU-readable and ready in the selected CUDA context while
    /// also satisfying Rust slice requirements. Boundary order and values are not checked;
    /// the wrapper does not check boundary storage contexts or wait for its producers.
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
    /// Each boundary array must have at least two entries and a length representable as `i32`.
    /// Each histogram output must contain exactly its level count minus one entries.
    pub fn histogram_range_channels_into<Level>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C3>,
        histograms: [&mut (impl DeviceOutput<i32> + ?Sized); 3],
        levels: [&[Level]; 3],
    ) -> Result<()>
    where
        Level: Copy,
        Self: HistogramRangeChannelsImage<T, C3, Level, 3>,
    {
        <Self as HistogramRangeChannelsImage<T, C3, Level, 3>>::histogram_range_channels(
            stream_context,
            source,
            histograms,
            levels,
        )
    }

    /// Computes per-channel histograms using caller-supplied bin boundaries.
    /// Processes all three C3 channels. Each channel has its own
    /// boundary array and histogram.
    /// Bin `k` counts samples satisfying `levels[k] <= sample < levels[k + 1]` for that channel.
    /// Each histogram contains exactly one fewer bin than its boundary array. The provided
    /// integer-image implementations use `i32` boundaries; `f32` images use `f32` boundaries.
    /// Passes boundary slice addresses directly to NPP as device pointers without uploading
    /// them. Storage must remain GPU-readable and ready in the selected CUDA context while
    /// also satisfying Rust slice requirements. Boundary order and values are not checked;
    /// the wrapper does not check boundary storage contexts or wait for its producers.
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
    /// Each boundary array must have at least two entries and a length representable as `i32`.
    /// Result allocation or source view construction can also fail.
    pub fn histogram_range_channels<Level>(self, levels: [&[Level]; 3]) -> Result<ImageHistograms>
    where
        Level: Copy,
        Self: HistogramRangeChannelsImage<T, C3, Level, 3>,
    {
        let [mut h0, mut h1, mut h2] =
            histogram_range_buffers(self.stream_context.cuda_context(), levels)?;
        {
            let source = self.view()?;
            <Self as HistogramRangeChannelsImage<T, C3, Level, 3>>::histogram_range_channels(
                self.stream_context,
                &source,
                [&mut h0, &mut h1, &mut h2],
                levels,
            )?;
        }
        // SAFETY: the histogram calls above queued a producer for every bin on this stream.
        unsafe { image_histograms(self.stream_context, vec![h0, h1, h2]) }
    }
}
