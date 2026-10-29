use singe_cuda::{
    memory::DeviceMemory,
    view::{DeviceOutput, DeviceOutputView},
};

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{ChannelLayout, ImageView},
    pipeline::image::{
        ImagePipeline,
        statistics::{HistogramEvenImage, HistogramRangeImage, ImageStatistic},
    },
};

mod count_in_range_methods;

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
{
    /// Computes a C1 histogram using caller-supplied bin boundaries.
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
    pub fn histogram_range_into<Level>(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        histogram: &mut (impl DeviceOutput<i32> + ?Sized),
        levels: &[Level],
    ) -> Result<()>
    where
        Level: Copy,
        Self: HistogramRangeImage<T, L, Level>,
    {
        <Self as HistogramRangeImage<T, L, Level>>::histogram_range(
            stream_context,
            source,
            histogram,
            levels,
        )
    }

    /// Computes a C1 histogram using caller-supplied bin boundaries.
    /// Bin `k` counts samples satisfying `levels[k] <= sample < levels[k + 1]` for that channel.
    /// Each histogram contains exactly one fewer bin than its boundary array. The provided
    /// integer-image implementations use `i32` boundaries; `f32` images use `f32` boundaries.
    /// Passes boundary slice addresses directly to NPP as device pointers without uploading
    /// them. Storage must remain GPU-readable and ready in the selected CUDA context while
    /// also satisfying Rust slice requirements. Boundary order and values are not checked;
    /// the wrapper does not check boundary storage contexts or wait for its producers.
    /// Allocates 1 owned device histogram containing the corresponding `i32` bin counts.
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
    pub fn histogram_range<Level>(self, levels: &[Level]) -> Result<ImageStatistic<i32>>
    where
        Level: Copy,
        Self: HistogramRangeImage<T, L, Level>,
    {
        self.ensure_eager()?;
        let mut histogram = DeviceMemory::<i32>::create_uninit(
            self.stream_context.cuda_context(),
            histogram_range_bins(levels)?,
        )?;
        {
            let source = self.view()?;
            <Self as HistogramRangeImage<T, L, Level>>::histogram_range(
                self.stream_context,
                &source,
                &mut DeviceOutputView::from_output(&mut histogram),
                levels,
            )?;
        }
        // SAFETY: the completed operation above wrote every element.
        Ok(ImageStatistic::from_values(unsafe {
            histogram.assume_init()
        }))
    }
}

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: HistogramEvenImage<T, L>,
{
    /// Computes a C1 histogram using evenly spaced bins.
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
    pub fn histogram_even_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        histogram: &mut (impl DeviceOutput<i32> + ?Sized),
        levels: i32,
        lower_level: i32,
        upper_level: i32,
    ) -> Result<()> {
        <Self as HistogramEvenImage<T, L>>::histogram_even(
            stream_context,
            source,
            histogram,
            levels,
            lower_level,
            upper_level,
        )
    }

    /// Computes a C1 histogram using evenly spaced bins.
    /// The lower bound is inclusive and the upper bound is exclusive. Each level count
    /// specifies one more boundary than bins, yielding `levels - 1` bins per channel. The
    /// wrapper forwards bounds to NPP without validating their order.
    /// Allocates 1 owned device histogram containing the corresponding `i32` bin counts.
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
    pub fn histogram_even(
        self,
        levels: i32,
        lower_level: i32,
        upper_level: i32,
    ) -> Result<ImageStatistic<i32>> {
        self.ensure_eager()?;
        if levels < 2 {
            return Err(Error::out_of_range("levels"));
        }
        let bins = usize::try_from(levels - 1).map_err(|_| Error::out_of_range("levels"))?;
        let mut histogram =
            DeviceMemory::<i32>::create_uninit(self.stream_context.cuda_context(), bins)?;
        {
            let source = self.view()?;
            <Self as HistogramEvenImage<T, L>>::histogram_even(
                self.stream_context,
                &source,
                &mut DeviceOutputView::from_output(&mut histogram),
                levels,
                lower_level,
                upper_level,
            )?;
        }
        // SAFETY: the completed operation above wrote every element.
        Ok(ImageStatistic::from_values(unsafe {
            histogram.assume_init()
        }))
    }
}

fn histogram_range_bins<Level>(levels: &[Level]) -> Result<usize> {
    if levels.len() < 2 {
        return Err(Error::out_of_range("levels"));
    }
    Ok(levels.len() - 1)
}
