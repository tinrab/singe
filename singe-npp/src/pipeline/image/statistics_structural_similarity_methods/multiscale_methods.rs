use singe_cuda::{memory::DeviceMemory, view::DeviceOutputView};

use crate::{
    context::StreamContext,
    error::Result,
    image::view::{ChannelLayout, ImageView},
    pipeline::image::{
        ImagePipeline,
        statistics::{
            ImageStatistic, MultiscaleStructuralSimilarityImage,
            WeightedMultiscaleStructuralSimilarityImage,
        },
    },
};

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: MultiscaleStructuralSimilarityImage<T, L>,
{
    /// Computes native multiscale structural similarity (MSSSIM) between two image ROIs.
    /// The provided C1 implementations produce one `f32` result.
    /// The supplied output view must contain exactly one element.
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
    pub fn multiscale_structural_similarity_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f32>,
    ) -> Result<()> {
        <Self as MultiscaleStructuralSimilarityImage<T, L>>::multiscale_structural_similarity(
            stream_context,
            source_1,
            source_2,
            output,
        )
    }

    /// Computes native multiscale structural similarity (MSSSIM) between two image ROIs.
    /// The provided C1 implementations produce one `f32` result.
    /// Allocates owned device storage for the single result.
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
    pub fn multiscale_structural_similarity(
        self,
        other: &ImageView<'_, T, L>,
    ) -> Result<ImageStatistic<f32>> {
        self.ensure_eager()?;
        let mut output = DeviceMemory::<f32>::create_uninit(
            self.stream_context.cuda_context(),
            <Self as MultiscaleStructuralSimilarityImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            <Self as MultiscaleStructuralSimilarityImage<T, L>>::multiscale_structural_similarity(
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

impl<T, L> ImagePipeline<'_, T, L>
where
    T: Copy,
    L: ChannelLayout,
    Self: WeightedMultiscaleStructuralSimilarityImage<T, L>,
{
    /// Computes native weighted multiscale structural similarity (WMSSSIM) between two image ROIs.
    /// Uses the native weighting and scale configuration.
    /// The provided C1, C3, and C4 implementations produce one, three, or four `f32` results respectively.
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
    pub fn weighted_multiscale_structural_similarity_into(
        stream_context: &StreamContext,
        source_1: &ImageView<'_, T, L>,
        source_2: &ImageView<'_, T, L>,
        output: &mut DeviceOutputView<'_, f32>,
    ) -> Result<()> {
        <Self as WeightedMultiscaleStructuralSimilarityImage<
            T,
            L,
        >>::weighted_multiscale_structural_similarity(
            stream_context,
            source_1,
            source_2,
            output,
        )
    }

    /// Computes native weighted multiscale structural similarity (WMSSSIM) between two image ROIs.
    /// Uses the native weighting and scale configuration.
    /// The provided C1, C3, and C4 implementations produce one, three, or four `f32` results respectively.
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
    pub fn weighted_multiscale_structural_similarity(
        self,
        other: &ImageView<'_, T, L>,
    ) -> Result<ImageStatistic<f32>> {
        self.ensure_eager()?;
        let mut output = DeviceMemory::<f32>::create_uninit(
            self.stream_context.cuda_context(),
            <Self as WeightedMultiscaleStructuralSimilarityImage<T, L>>::OUTPUT_CHANNELS,
        )?;
        {
            let source = self.view()?;
            <Self as WeightedMultiscaleStructuralSimilarityImage<
                T,
                L,
            >>::weighted_multiscale_structural_similarity(
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
