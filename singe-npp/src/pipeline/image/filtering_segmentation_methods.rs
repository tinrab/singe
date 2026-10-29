use crate::{
    context::StreamContext,
    error::Result,
    image::view::{C1, ImageOutputView, ImageView},
    pipeline::image::filtering_segmentation_traits::{
        LabelMarkersUfBatchImage, LabelMarkersUfImage,
    },
    types::ImageNormalization,
};

mod contours_methods;
mod distance_methods;
mod marker_compression_methods;

use crate::pipeline::image::{ContiguousImage, ImagePipeline, MarkerLabels};

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
    Self: LabelMarkersUfImage<T>,
{
    /// Labels connected equal-valued regions with the native union-find algorithm.
    ///
    /// The provided implementations accept C1 `u8`, `u16`, or `u32` images and produce
    /// C1 `u32` marker IDs. `ImageNormalization::L1` selects four-way connectivity;
    /// `ImageNormalization::Infinity` selects eight-way connectivity. NPP may reject other
    /// normalizations. IDs can have gaps and have no defined ordering.
    /// Source and destination dimensions must match. The destination must use a linear CUDA
    /// allocation with a byte stride of source width times `size_of::<u32>()`; the wrapper
    /// does not enforce this native allocation or stride requirement. Owns temporary device
    /// scratch through native completion.
    /// NPP receives a mutable source pointer; source storage must be exclusively available
    /// during native use. Row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Native connectivity restrictions and checked byte/count conversions can also fail.
    /// Scratch-size queries and allocation can fail.
    /// Source and destination dimensions must match.
    pub fn label_markers_uf_into(
        stream_context: &StreamContext,
        source: &ImageView<'_, T, C1>,
        destination: &mut ImageOutputView<'_, u32, C1>,
        norm: ImageNormalization,
    ) -> Result<()> {
        <Self as LabelMarkersUfImage<T>>::label_markers_uf_image(
            stream_context,
            source,
            destination,
            norm,
        )
    }
}

impl<T> ImagePipeline<'_, T, C1>
where
    T: Copy,
{
    /// Labels connected equal-valued regions with the native union-find algorithm.
    ///
    /// The provided implementations accept C1 `u8`, `u16`, or `u32` images and produce
    /// C1 `u32` marker IDs. `ImageNormalization::L1` selects four-way connectivity;
    /// `ImageNormalization::Infinity` selects eight-way connectivity. NPP may reject other
    /// normalizations. IDs can have gaps and have no defined ordering.
    /// Uses the smallest width and height across the batch as a common top-left ROI.
    /// Destination pixels outside that ROI remain unwritten.
    /// Destinations must use linear CUDA allocations with a byte stride of the common ROI
    /// width times `size_of::<u32>()`; the wrapper does not enforce this native requirement.
    /// Uploads temporary device descriptors and retains them through native completion.
    /// NPP receives a mutable source pointer; source storage must be exclusively available
    /// during native use. Row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Native connectivity restrictions and checked byte/count conversions can also fail.
    /// Batches must have equal lengths of at least two, representable as `i32`; each source
    /// and destination pair must have equal dimensions. Descriptor allocation or upload can fail.
    pub fn label_markers_uf_batch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, C1>],
        destinations: &mut [ImageOutputView<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()>
    where
        Self: LabelMarkersUfBatchImage<T>,
    {
        <Self as LabelMarkersUfBatchImage<T>>::label_markers_uf_batch_image(
            stream_context,
            sources,
            destinations,
            norm,
        )
    }

    /// Labels connected equal-valued regions with the native union-find algorithm.
    ///
    /// The provided implementations accept C1 `u8`, `u16`, or `u32` images and produce
    /// C1 `u32` marker IDs. `ImageNormalization::L1` selects four-way connectivity;
    /// `ImageNormalization::Infinity` selects eight-way connectivity. NPP may reject other
    /// normalizations. IDs can have gaps and have no defined ordering.
    /// Each source/destination pair uses its own size; the largest destination width and
    /// height determine the native launch bounds.
    /// Destinations must use linear CUDA allocations with a byte stride of their own
    /// width times `size_of::<u32>()`; the wrapper does not enforce this native requirement.
    /// Uploads temporary device descriptors and retains them through native completion.
    /// NPP receives a mutable source pointer; source storage must be exclusively available
    /// during native use. Row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Native connectivity restrictions and checked byte/count conversions can also fail.
    /// Batches must have equal lengths of at least two, representable as `i32`; each source
    /// and destination pair must have equal dimensions. Descriptor allocation or upload can fail.
    pub fn label_markers_uf_batch_advanced(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, T, C1>],
        destinations: &mut [ImageOutputView<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()>
    where
        Self: LabelMarkersUfBatchImage<T>,
    {
        <Self as LabelMarkersUfBatchImage<T>>::label_markers_uf_batch_advanced_image(
            stream_context,
            sources,
            destinations,
            norm,
        )
    }

    /// Labels connected equal-valued regions with the native union-find algorithm.
    ///
    /// The provided implementations accept C1 `u8`, `u16`, or `u32` images and produce
    /// C1 `u32` marker IDs. `ImageNormalization::L1` selects four-way connectivity;
    /// `ImageNormalization::Infinity` selects eight-way connectivity. NPP may reject other
    /// normalizations. IDs can have gaps and have no defined ordering.
    /// Allocates a new source-sized, contiguous `u32` image and temporary device scratch.
    /// Returns the unchanged native label grid as `MarkerLabels` for later compression.
    /// NPP receives a mutable source pointer; source storage must be exclusively available
    /// during native use. Row padding remains unwritten.
    /// Prepares image regions and waits for producers on other streams. Rejects stream
    /// capture and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, native execution, or stream synchronization fail.
    /// Native connectivity restrictions and checked byte/count conversions can also fail.
    /// Scratch-size queries and allocation can fail.
    /// Output allocation or view construction can also fail.
    pub fn label_markers_uf(self, norm: ImageNormalization) -> Result<MarkerLabels>
    where
        Self: LabelMarkersUfImage<T>,
    {
        self.ensure_eager()?;
        let mut destination = ContiguousImage::<u32, C1>::create_uninit(
            self.stream_context.cuda_context(),
            self.size(),
        )?;

        {
            let source = self.view()?;
            let mut destination_view = destination.output_view()?;
            <Self as LabelMarkersUfImage<T>>::label_markers_uf_image(
                self.stream_context,
                &source,
                &mut destination_view,
                norm,
            )?;
        }

        // SAFETY: completed labeling produced this unchanged dense native grid.
        Ok(unsafe { MarkerLabels::from_image(destination.assume_init()) })
    }
}
