use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

mod error_metric_methods;
mod quality_metric_methods;

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
    pipeline::image::{
        ImagePipeline,
        statistics::{DotProductImage, ImageStatistic},
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: DotProductImage<T, L>,
{
    /// Computes native per-channel dot products between two image ROIs.
    /// Produces one `f64` sum of corresponding sample products per processed channel: one
    /// for C1, three for C3 and AC4, or four for C4. AC4 alpha is excluded; C4 includes its
    /// fourth channel.
    /// The supplied output view must contain exactly the result count described above.
    /// Prepares both source images and output storage, waits for producers on other
    /// streams, rejects capture, queries and allocates temporary device scratch, and returns
    /// after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native execution, or stream synchronization fail.
    /// Source image dimensions must match.
    /// The output view must have the exact result count described above.
    pub fn dot_product_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f64>,
    ) -> Result<()> {
        <Self as DotProductImage<T, L>>::dot_product(stream_context, source_1, source_2, output)
    }

    /// Computes native per-channel dot products between two image ROIs.
    /// Produces one `f64` sum of corresponding sample products per processed channel: one
    /// for C1, three for C3 and AC4, or four for C4. AC4 alpha is excluded; C4 includes its
    /// fourth channel.
    /// Allocates owned device storage for the complete result array.
    /// Prepares both source images and output storage, waits for producers on other
    /// streams, rejects capture, queries and allocates temporary device scratch, and returns
    /// after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region validation, producer
    /// waits, scratch-size queries or allocation, native execution, or stream synchronization fail.
    /// Source image dimensions must match.
    /// Result allocation or source view construction can also fail.
    pub fn dot_product(self, other: &ImageView<'_, T, L>) -> Result<ImageStatistic<f64>> {
        self.ensure_eager()?;
        let mut output = DeviceMemory::<f64>::create_uninit(
            self.stream_context.cuda_context(),
            <Self as DotProductImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            <Self as DotProductImage<T, L>>::dot_product(
                self.stream_context,
                &source,
                other,
                &mut DeviceOutputView::from_output(&mut output),
            )?;
        }
        // SAFETY: the completed operation above wrote every element.
        Ok(ImageStatistic::from_values(unsafe { output.assume_init() }))
    }
}
