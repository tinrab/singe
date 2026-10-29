use singe_cuda::memory::DeviceMemory;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        filtering,
        view::{C1, ImageViewMut},
    },
    pipeline::image::ImagePipeline,
};

impl ImagePipeline<'_, u32, C1> {
    /// Renumbers union-find marker IDs in place to remove numbering gaps.
    ///
    /// Returns an owned device array containing one new maximum label ID per image.
    /// All images use a common size and equal per-image scratch capacity.
    /// Owns temporary device descriptors and scratch through native completion. Prepares
    /// image and result regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, checked ROI-area or byte-size arithmetic, scratch queries, allocation,
    /// descriptor uploads, native compression, or synchronization fail. The batch must
    /// contain at least two images and fit `i32`. Each stride must equal its image width
    /// times `size_of::<u32>()`. All image dimensions must match.
    ///
    /// # Safety
    ///
    /// Each view must exclusively retain its original ordinary union-find allocation and
    /// exact ROI from the matching completed native label-generation call. Use linear CUDA
    /// allocations with the same base pointer and ROI, rather than pitched allocations.
    /// Keep all images disjoint and exclusively available during native use.
    pub unsafe fn compress_marker_labels_uf_batch(
        stream_context: &StreamContext,
        source_destinations: &mut [ImageViewMut<'_, u32, C1>],
    ) -> Result<DeviceMemory<u32>> {
        stream_context.stream().ensure_not_capturing()?;
        let mut new_max_label_ids = DeviceMemory::<u32>::create_uninit(
            stream_context.cuda_context(),
            source_destinations.len(),
        )?;
        for image in source_destinations.iter() {
            stream_context.prepare_region(image)?;
        }
        stream_context.prepare_region(&new_max_label_ids)?;
        // SAFETY: the native helper completes borrowed use on every path.
        unsafe {
            filtering::compress_marker_labels_uf_batch_u32_c1_in_place(
                stream_context,
                source_destinations,
                &mut new_max_label_ids,
            )
        }?;
        // SAFETY: completed compression writes one count for each batch entry.
        Ok(unsafe { new_max_label_ids.assume_init() })
    }

    /// Renumbers union-find marker IDs in place to remove numbering gaps.
    ///
    /// Returns an owned device array containing one new maximum label ID per image.
    /// Uses each image's dimensions and scratch size; the largest width, height, and
    /// scratch size determine native launch bounds.
    /// Owns temporary device descriptors and scratch through native completion. Prepares
    /// image and result regions, waits for producers on other streams, rejects capture,
    /// and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if capture checks, CUDA context binding or region preparation,
    /// producer waits, checked ROI-area or byte-size arithmetic, scratch queries, allocation,
    /// descriptor uploads, native compression, or synchronization fail. The batch must
    /// contain at least two images and fit `i32`. Each stride must equal its image width
    /// times `size_of::<u32>()`.
    ///
    /// # Safety
    ///
    /// Each view must exclusively retain its original ordinary union-find allocation and
    /// exact ROI from the matching completed native label-generation call. Use linear CUDA
    /// allocations with the same base pointer and ROI, rather than pitched allocations.
    /// Keep all images disjoint and exclusively available during native use.
    pub unsafe fn compress_marker_labels_uf_batch_advanced(
        stream_context: &StreamContext,
        source_destinations: &mut [ImageViewMut<'_, u32, C1>],
    ) -> Result<DeviceMemory<u32>> {
        stream_context.stream().ensure_not_capturing()?;
        let mut new_max_label_ids = DeviceMemory::<u32>::create_uninit(
            stream_context.cuda_context(),
            source_destinations.len(),
        )?;
        for image in source_destinations.iter() {
            stream_context.prepare_region(image)?;
        }
        stream_context.prepare_region(&new_max_label_ids)?;
        // SAFETY: the native helper completes borrowed use on every path.
        unsafe {
            filtering::compress_marker_labels_uf_batch_u32_c1_in_place_advanced(
                stream_context,
                source_destinations,
                &mut new_max_label_ids,
            )
        }?;
        // SAFETY: completed compression writes one count for each batch entry.
        Ok(unsafe { new_max_label_ids.assume_init() })
    }
}
