use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView, MaskView},
    pipeline::image::{
        statistics::{
            ImageMeanStandardDeviation, ImageStatistic, MaskedMeanStandardDeviationImage,
        },
        statistics_masked_methods::ImagePipeline,
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MaskedMeanStandardDeviationImage<T, L>,
{
    /// Computes the native mean and standard deviation over pixels selected by `mask`.
    /// Only nonzero mask samples select pixels. Image and mask dimensions must match.
    /// The provided implementations process single-channel images and produce one `f64`
    /// result for each of the mean and standard deviation.
    /// Each supplied output view must contain exactly one element.
    /// Prepares all participating images, masks, and outputs, waits for producers on other
    /// streams, rejects capture, and returns after successful native completion.
    /// The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Each output view must have length one.
    /// Image and mask dimensions must match. Scratch-size queries and allocation can also fail.
    pub fn mean_standard_deviation_masked_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, L>,
        mask: &MaskView<'_>,
        mean: &mut DeviceOutputView<'_, f64>,
        standard_deviation: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as MaskedMeanStandardDeviationImage<T, L>>::mean_standard_deviation_masked(
            stream_context,
            source,
            mask,
            mean,
            standard_deviation,
        )
    }

    /// Computes the native mean and standard deviation over pixels selected by `mask`.
    /// Only nonzero mask samples select pixels. Image and mask dimensions must match.
    /// The provided implementations process single-channel images and produce one `f64`
    /// result for each of the mean and standard deviation.
    /// Allocates owned device result storage.
    /// Prepares all participating images, masks, and outputs, waits for producers on other
    /// streams, rejects capture, and returns after successful native completion.
    /// The native reduction owns its temporary scratch.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, native execution, or stream synchronization fail.
    /// Result allocation and constructing the source view can also fail.
    /// Image and mask dimensions must match. Scratch-size queries and allocation can also fail.
    pub fn mean_standard_deviation_masked(
        self,
        mask: &MaskView<'_>,
    ) -> Result<ImageMeanStandardDeviation<f64>> {
        self.ensure_eager()?;
        let mut mean = DeviceMemory::<f64>::create_uninit(
            self.stream_context.cuda_context(),
            <Self as MaskedMeanStandardDeviationImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        let mut standard_deviation = DeviceMemory::<f64>::create_uninit(
            self.stream_context.cuda_context(),
            <Self as MaskedMeanStandardDeviationImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            <Self as MaskedMeanStandardDeviationImage<T, L>>::mean_standard_deviation_masked(
                self.stream_context,
                &source,
                mask,
                &mut DeviceOutputView::from_output(&mut mean),
                &mut DeviceOutputView::from_output(&mut standard_deviation),
            )?;
        }
        Ok(ImageMeanStandardDeviation {
            // SAFETY: the completed operation above wrote every element.
            mean: ImageStatistic::from_values(unsafe { mean.assume_init() }),
            // SAFETY: the completed operation above wrote every element.
            standard_deviation: ImageStatistic::from_values(unsafe {
                standard_deviation.assume_init()
            }),
        })
    }
}
