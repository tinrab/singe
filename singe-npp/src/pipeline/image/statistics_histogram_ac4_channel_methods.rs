use singe_cuda::view::DeviceOutput;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{AC4, ImageView},
    pipeline::image::{
        ImagePipeline,
        statistics::{HistogramEvenChannelsImage, HistogramRangeChannelsImage, ImageHistograms},
        statistics_histogram_channel_methods::{
            histogram_buffers, histogram_range_buffers, image_histograms,
        },
    },
};

impl<T> ImagePipeline<'_, T, AC4>
where
    T: Copy,
    Self: HistogramEvenChannelsImage<T, AC4, 3>,
{
    /// Computes per-channel histograms using evenly spaced bins.
    /// Processes the first three AC4 channels and excludes alpha. Each channel has its own
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
        source: &ImageView<'_, T, AC4>,
        histograms: [&mut (impl DeviceOutput<i32> + ?Sized); 3],
        levels: [i32; 3],
        lower_levels: [i32; 3],
        upper_levels: [i32; 3],
    ) -> Result<()> {
        <Self as HistogramEvenChannelsImage<T, AC4, 3>>::histogram_even_channels(
            stream_context,
            source,
            histograms,
            levels,
            lower_levels,
            upper_levels,
        )
    }

    /// Computes per-channel histograms using evenly spaced bins.
    /// Processes the first three AC4 channels and excludes alpha. Each channel has its own
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
            <Self as HistogramEvenChannelsImage<T, AC4, 3>>::histogram_even_channels(
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

impl<T> ImagePipeline<'_, T, AC4>
where
    T: Copy,
{
    /// Computes per-channel histograms using caller-supplied bin boundaries.
    /// Processes the first three AC4 channels and excludes alpha. Each channel has its own
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
        source: &ImageView<'_, T, AC4>,
        histograms: [&mut (impl DeviceOutput<i32> + ?Sized); 3],
        levels: [&[Level]; 3],
    ) -> Result<()>
    where
        Level: Copy,
        Self: HistogramRangeChannelsImage<T, AC4, Level, 3>,
    {
        <Self as HistogramRangeChannelsImage<T, AC4, Level, 3>>::histogram_range_channels(
            stream_context,
            source,
            histograms,
            levels,
        )
    }

    /// Computes per-channel histograms using caller-supplied bin boundaries.
    /// Processes the first three AC4 channels and excludes alpha. Each channel has its own
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
        Self: HistogramRangeChannelsImage<T, AC4, Level, 3>,
    {
        let [mut h0, mut h1, mut h2] =
            histogram_range_buffers(self.stream_context.cuda_context(), levels)?;
        {
            let source = self.view()?;
            <Self as HistogramRangeChannelsImage<T, AC4, Level, 3>>::histogram_range_channels(
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
