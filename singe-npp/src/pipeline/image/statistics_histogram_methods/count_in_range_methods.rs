use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
    pipeline::image::{
        statistics::{CountInRangeImage, ImageStatistic},
        statistics_histogram_methods::ImagePipeline,
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: CountInRangeImage<T, L>,
{
    /// Counts samples within each processed channel intensity range, including both bounds.
    /// The provided C1 implementation uses scalar bounds and one `i32` result; C3 and AC4
    /// use three bounds per side and three results. AC4 alpha is excluded. Bounds are forwarded
    /// to NPP without checking their order or floating-point values.
    /// The supplied counts view must have exactly the number of processed channels.
    /// Prepares source and result regions and waits for producers on other streams. Rejects
    /// capture, queries and allocates temporary device scratch, and returns after successful
    /// stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or source/result region validation,
    /// producer waits, scratch-size queries or allocation, native execution, or stream
    /// synchronization fail.
    /// The counts view must have the exact processed-channel count.
    pub fn count_in_range_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        counts: &mut DeviceOutputView<'_, i32>,
        lower_bound: <Self as CountInRangeImage<T, L>>::Bounds,
        upper_bound: <Self as CountInRangeImage<T, L>>::Bounds,
    ) -> Result<()> {
        <Self as CountInRangeImage<T, L>>::count_in_range(
            stream_context,
            source,
            counts,
            lower_bound,
            upper_bound,
        )
    }

    /// Counts samples within each processed channel intensity range, including both bounds.
    /// The provided C1 implementation uses scalar bounds and one `i32` result; C3 and AC4
    /// use three bounds per side and three results. AC4 alpha is excluded. Bounds are forwarded
    /// to NPP without checking their order or floating-point values.
    /// Allocates owned device storage for the per-channel `i32` counts.
    /// Prepares source and result regions and waits for producers on other streams. Rejects
    /// capture, queries and allocates temporary device scratch, and returns after successful
    /// stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or source/result region validation,
    /// producer waits, scratch-size queries or allocation, native execution, or stream
    /// synchronization fail.
    /// Result allocation or source view construction can also fail.
    pub fn count_in_range(
        self,
        lower_bound: <Self as CountInRangeImage<T, L>>::Bounds,
        upper_bound: <Self as CountInRangeImage<T, L>>::Bounds,
    ) -> Result<ImageStatistic<i32>> {
        self.ensure_eager()?;
        let mut counts = DeviceMemory::<i32>::create_uninit(
            self.stream_context.cuda_context(),
            <Self as CountInRangeImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            <Self as CountInRangeImage<T, L>>::count_in_range(
                self.stream_context,
                &source,
                &mut DeviceOutputView::from_output(&mut counts),
                lower_bound,
                upper_bound,
            )?;
        }
        // SAFETY: the completed operation above wrote every element.
        Ok(ImageStatistic::from_values(unsafe { counts.assume_init() }))
    }
}
