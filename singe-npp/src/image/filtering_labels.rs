use std::ptr;

use singe_cuda::{memory::DeviceMemory, view::DeviceOutput};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        filtering_validation::{
            compress_marker_labels_uf_batch_advanced_buffers,
            compress_marker_labels_uf_batch_advanced_descriptors,
            compress_marker_labels_uf_batch_buffers, compress_marker_labels_uf_batch_descriptors,
            label_markers_uf_batch_advanced_descriptors, label_markers_uf_batch_descriptors,
            roi_area, validate_batch_label_output, validate_block_segment_inputs,
            validate_contours_geometry_image_host, validate_geometry_list_buffers,
            validate_geometry_list_id_range, validate_marker_label_info_host_output,
            validate_marker_label_info_output, validate_same_size, validate_typed_output_len,
            validate_u32_label_marker_step, validate_u32_list_dev_output,
            validate_u32_list_device_and_host_output, validate_u32_list_host_output,
            validate_u32_list_output,
        },
        view::{C1, ImageOutputView, ImageView, ImageViewMut},
    },
    try_ffi,
    types::{
        CompressedMarkerLabelsInfo, ContourBlockSegment, ContourPixelDirectionInfo,
        ContourPixelGeometryInfo, ContourTotalsInfo, DataTypeLike, ImageNormalization, Point32f,
        Point64f, Size,
    },
    utility::to_usize,
};

/// Returns the required scratch capacity in bytes for union-find labeling of `roi`.
///
/// The provided query forwards the ROI dimensions to NPP.
///
/// # Errors
///
/// Returns an error if NPP rejects the query or the returned byte count cannot be represented as `usize`.
pub fn label_markers_uf_buffer_size_u32_c1(roi: Size) -> Result<usize> {
    let mut bytes = 0;
    // SAFETY: `nppiLabelMarkersUFGetBufferSize_32u_C1R` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiLabelMarkersUFGetBufferSize_32u_C1R(
            roi.into(),
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

/// Labels connected regions with the native union-find algorithm.
///
/// A connected region contains pixels with the same value. `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity.
/// Native marker IDs may have gaps and do not have a defined ordering.
/// The provided implementations reject stream capture and return after successful stream completion.
/// Temporary device scratch or descriptor allocations remain alive through native use.
///
/// # Errors
///
/// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
/// Scratch-size queries, device allocations or descriptor uploads, capture checks, and stream synchronization can also fail.
/// Source and destination image sizes must match.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Inputs must be initialized and ready for native execution.
/// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
/// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
/// NPP receives mutable source pointers; provide exclusive access to source storage during native use.
/// Retain all allocations until native work completes.
/// Native UF destinations must use linear CUDA allocations, rather than pitched allocations, with a byte stride of the source image width times `size_of::<u32>()`. The wrapper does not enforce this allocation or stride requirement.
pub unsafe fn label_markers_uf_u8_to_u32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u8, C1>,
    destination: &mut ImageOutputView<'_, u32, C1>,
    norm: ImageNormalization,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.stream().ensure_not_capturing()?;

    validate_same_size(source.size(), destination.size())?;
    let required_bytes = label_markers_uf_buffer_size_u32_c1(source.size())?;
    let scratch = DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

    // SAFETY: `nppiLabelMarkersUF_8u32u_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    stream_context.sync_scope(|| unsafe {
        try_ffi!(sys::nppiLabelMarkersUF_8u32u_C1R_Ctx(
            source.as_ptr().cast_mut().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            norm.into(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))
    })?;

    Ok(())
}

/// Labels connected regions with the native union-find algorithm.
///
/// A connected region contains pixels with the same value. `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity.
/// Native marker IDs may have gaps and do not have a defined ordering.
/// The provided implementations reject stream capture and return after successful stream completion.
/// Temporary device scratch or descriptor allocations remain alive through native use.
///
/// # Errors
///
/// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
/// Scratch-size queries, device allocations or descriptor uploads, capture checks, and stream synchronization can also fail.
/// Source and destination image sizes must match.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Inputs must be initialized and ready for native execution.
/// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
/// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
/// NPP receives mutable source pointers; provide exclusive access to source storage during native use.
/// Retain all allocations until native work completes.
/// Native UF destinations must use linear CUDA allocations, rather than pitched allocations, with a byte stride of the source image width times `size_of::<u32>()`. The wrapper does not enforce this allocation or stride requirement.
pub unsafe fn label_markers_uf_u16_to_u32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u16, C1>,
    destination: &mut ImageOutputView<'_, u32, C1>,
    norm: ImageNormalization,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.stream().ensure_not_capturing()?;

    validate_same_size(source.size(), destination.size())?;
    let required_bytes = label_markers_uf_buffer_size_u32_c1(source.size())?;
    let scratch = DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

    // SAFETY: `nppiLabelMarkersUF_16u32u_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    stream_context.sync_scope(|| unsafe {
        try_ffi!(sys::nppiLabelMarkersUF_16u32u_C1R_Ctx(
            source.as_ptr().cast_mut().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            norm.into(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))
    })?;

    Ok(())
}

/// Labels connected regions with the native union-find algorithm.
///
/// A connected region contains pixels with the same value. `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity.
/// Native marker IDs may have gaps and do not have a defined ordering.
/// The provided implementations reject stream capture and return after successful stream completion.
/// Temporary device scratch or descriptor allocations remain alive through native use.
///
/// # Errors
///
/// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
/// Scratch-size queries, device allocations or descriptor uploads, capture checks, and stream synchronization can also fail.
/// Source and destination image sizes must match.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Inputs must be initialized and ready for native execution.
/// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
/// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
/// NPP receives mutable source pointers; provide exclusive access to source storage during native use.
/// Retain all allocations until native work completes.
/// Native UF destinations must use linear CUDA allocations, rather than pitched allocations, with a byte stride of the source image width times `size_of::<u32>()`. The wrapper does not enforce this allocation or stride requirement.
pub unsafe fn label_markers_uf_u32_c1(
    stream_context: &StreamContext,
    source: &ImageView<'_, u32, C1>,
    destination: &mut ImageOutputView<'_, u32, C1>,
    norm: ImageNormalization,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(source)?;
    stream_context.validate_region(destination)?;
    stream_context.stream().ensure_not_capturing()?;

    validate_same_size(source.size(), destination.size())?;
    let required_bytes = label_markers_uf_buffer_size_u32_c1(source.size())?;
    let scratch = DeviceMemory::<u8>::create_uninit(stream_context.cuda_context(), required_bytes)?;

    // SAFETY: `nppiLabelMarkersUF_32u_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    stream_context.sync_scope(|| unsafe {
        try_ffi!(sys::nppiLabelMarkersUF_32u_C1R_Ctx(
            source.as_ptr().cast_mut().cast(),
            source.step(),
            destination.as_mut_ptr().cast(),
            destination.step(),
            source.size().into(),
            norm.into(),
            scratch.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))
    })?;

    Ok(())
}

macro_rules! impl_label_markers_uf_batch {
    ($name:ident, $source_ty:ty, $ffi_name:ident) => {
        /// Labels connected regions with the native union-find algorithm.
        ///
        /// A connected region contains pixels with the same value. `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity.
        /// Native marker IDs may have gaps and do not have a defined ordering.
        /// The provided implementations reject stream capture and return after successful stream completion.
        /// Temporary device scratch or descriptor allocations remain alive through native use.
        /// The common ROI uses the smallest source or destination width and height in the batch. Pixels outside that ROI remain unwritten.
        ///
        /// # Errors
        ///
        /// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
        /// Scratch-size queries, device allocations or descriptor uploads, capture checks, and stream synchronization can also fail.
        /// Source and destination batches must have equal lengths of at least two, representable as `i32`. Each source and destination pair must have equal sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Inputs must be initialized and ready for native execution.
        /// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
        /// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
        /// NPP receives mutable source pointers; provide exclusive access to source storage during native use.
        /// Retain all allocations until native work completes.
        /// Native UF destinations must use linear CUDA allocations, rather than pitched allocations, with a byte stride of the common ROI width times `size_of::<u32>()`. The wrapper does not enforce this allocation or stride requirement.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, $source_ty, C1>],
            destinations: &mut [ImageOutputView<'_, u32, C1>],
            norm: ImageNormalization,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in sources.iter() {
                stream_context.validate_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, roi, source_descriptors, destination_descriptors) =
                label_markers_uf_batch_descriptors(
                    stream_context.cuda_context(),
                    sources,
                    destinations,
                )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi_name(
                    source_descriptors.as_ptr().cast(),
                    destination_descriptors.as_ptr().cast_mut().cast(),
                    batch_size,
                    roi.into(),
                    norm.into(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

macro_rules! impl_label_markers_uf_batch_advanced {
    ($name:ident, $source_ty:ty, $ffi_name:ident) => {
        /// Labels connected regions with the native union-find algorithm.
        ///
        /// A connected region contains pixels with the same value. `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity.
        /// Native marker IDs may have gaps and do not have a defined ordering.
        /// The provided implementations reject stream capture and return after successful stream completion.
        /// Temporary device scratch or descriptor allocations remain alive through native use.
        /// Each pair uses its own image size. The largest destination width and height determine the native launch bounds.
        ///
        /// # Errors
        ///
        /// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
        /// Scratch-size queries, device allocations or descriptor uploads, capture checks, and stream synchronization can also fail.
        /// Source and destination batches must have equal lengths of at least two, representable as `i32`. Each source and destination pair must have equal sizes.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Inputs must be initialized and ready for native execution.
        /// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
        /// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
        /// NPP receives mutable source pointers; provide exclusive access to source storage during native use.
        /// Retain all allocations until native work completes.
        /// Native UF destinations must use linear CUDA allocations, rather than pitched allocations, with a byte stride of each destination image width times `size_of::<u32>()`. The wrapper does not enforce this allocation or stride requirement.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            sources: &[ImageView<'_, $source_ty, C1>],
            destinations: &mut [ImageOutputView<'_, u32, C1>],
            norm: ImageNormalization,
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            for image in sources.iter() {
                stream_context.validate_region(image)?;
            }
            for image in destinations.iter() {
                stream_context.validate_region(image)?;
            }
            stream_context.stream().ensure_not_capturing()?;

            let (batch_size, max_roi, source_descriptors, destination_descriptors) =
                label_markers_uf_batch_advanced_descriptors(
                    stream_context.cuda_context(),
                    sources,
                    destinations,
                )?;

            // SAFETY: the caller upholds this function's safety contract.
            stream_context.sync_scope(|| unsafe {
                try_ffi!(sys::$ffi_name(
                    source_descriptors.as_ptr().cast(),
                    destination_descriptors.as_ptr().cast_mut().cast(),
                    batch_size,
                    max_roi.into(),
                    norm.into(),
                    stream_context.as_raw(),
                ))
            })?;

            Ok(())
        }
    };
}

impl_label_markers_uf_batch!(
    label_markers_uf_batch_u8_to_u32_c1,
    u8,
    nppiLabelMarkersUFBatch_8u32u_C1R_Ctx
);

impl_label_markers_uf_batch!(
    label_markers_uf_batch_u16_to_u32_c1,
    u16,
    nppiLabelMarkersUFBatch_16u32u_C1R_Ctx
);

impl_label_markers_uf_batch!(
    label_markers_uf_batch_u32_c1,
    u32,
    nppiLabelMarkersUFBatch_32u_C1R_Ctx
);

impl_label_markers_uf_batch_advanced!(
    label_markers_uf_batch_u8_to_u32_c1_advanced,
    u8,
    nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx
);

impl_label_markers_uf_batch_advanced!(
    label_markers_uf_batch_u16_to_u32_c1_advanced,
    u16,
    nppiLabelMarkersUFBatch_16u32u_C1R_Advanced_Ctx
);

impl_label_markers_uf_batch_advanced!(
    label_markers_uf_batch_u32_c1_advanced,
    u32,
    nppiLabelMarkersUFBatch_32u_C1R_Advanced_Ctx
);

/// Dispatches union-find connected-component labeling for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, connectivity, marker output semantics, and the documented execution contract.
/// Scratch-size queries must not underestimate native workspace requirements. Retain temporary resources through native completion.
pub unsafe trait LabelMarkersUfC1: DataTypeLike + Sized {
    /// Returns the required scratch capacity in bytes for union-find labeling of `roi`.
    ///
    /// The provided query forwards the ROI dimensions to NPP.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the query or the returned byte count cannot be represented as `usize`.
    fn buffer_size(roi: Size) -> Result<usize> {
        label_markers_uf_buffer_size_u32_c1(roi)
    }

    /// Labels connected regions with the native union-find algorithm.
    ///
    /// A connected region contains pixels with the same value. `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity.
    /// Native marker IDs may have gaps and do not have a defined ordering.
    /// The provided implementations reject stream capture and return after successful stream completion.
    /// Temporary device scratch or descriptor allocations remain alive through native use.
    ///
    /// # Errors
    ///
    /// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
    /// Scratch-size queries, device allocations or descriptor uploads, capture checks, and stream synchronization can also fail.
    /// Source and destination image sizes must match.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Inputs must be initialized and ready for native execution.
    /// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
    /// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
    /// NPP receives mutable source pointers; provide exclusive access to source storage during native use.
    /// Retain all allocations until native work completes.
    /// Native UF destinations must use linear CUDA allocations, rather than pitched allocations, with a byte stride of the source image width times `size_of::<u32>()`. The wrapper does not enforce this allocation or stride requirement.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, u32, C1>,
        norm: ImageNormalization,
    ) -> Result<()>;
}

unsafe impl LabelMarkersUfC1 for u8 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, u32, C1>,
        norm: ImageNormalization,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { label_markers_uf_u8_to_u32_c1(stream_context, source, destination, norm) }
    }
}

unsafe impl LabelMarkersUfC1 for u16 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, u32, C1>,
        norm: ImageNormalization,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { label_markers_uf_u16_to_u32_c1(stream_context, source, destination, norm) }
    }
}

unsafe impl LabelMarkersUfC1 for u32 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, u32, C1>,
        norm: ImageNormalization,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { label_markers_uf_u32_c1(stream_context, source, destination, norm) }
    }
}

/// Returns the required scratch capacity in bytes for union-find labeling of `roi`.
///
/// The provided query forwards the ROI dimensions to NPP.
///
/// # Errors
///
/// Returns an error if NPP rejects the query or the returned byte count cannot be represented as `usize`.
pub fn label_markers_uf_buffer_size_for<T: LabelMarkersUfC1>(roi: Size) -> Result<usize> {
    T::buffer_size(roi)
}

/// Labels connected regions with the native union-find algorithm.
///
/// A connected region contains pixels with the same value. `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity.
/// Native marker IDs may have gaps and do not have a defined ordering.
/// The provided implementations reject stream capture and return after successful stream completion.
/// Temporary device scratch or descriptor allocations remain alive through native use.
///
/// # Errors
///
/// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
/// Scratch-size queries, device allocations or descriptor uploads, capture checks, and stream synchronization can also fail.
/// Source and destination image sizes must match.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Inputs must be initialized and ready for native execution.
/// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
/// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
/// NPP receives mutable source pointers; provide exclusive access to source storage during native use.
/// Retain all allocations until native work completes.
/// Native UF destinations must use linear CUDA allocations, rather than pitched allocations, with a byte stride of the source image width times `size_of::<u32>()`. The wrapper does not enforce this allocation or stride requirement.
pub unsafe fn label_markers_uf<T: LabelMarkersUfC1>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, u32, C1>,
    norm: ImageNormalization,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::dispatch(stream_context, source, destination, norm) }
}

/// Dispatches union-find connected-component labeling for supported native element types.
///
/// # Safety
///
/// Implementations must preserve native element representations, connectivity, marker output semantics, and the documented execution contract.
/// Scratch-size queries must not underestimate native workspace requirements. Retain temporary resources through native completion.
pub unsafe trait LabelMarkersUfToC1<Destination>: DataTypeLike + Sized
where
    Destination: DataTypeLike,
{
    /// Labels connected regions with the native union-find algorithm.
    ///
    /// A connected region contains pixels with the same value. `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity.
    /// Native marker IDs may have gaps and do not have a defined ordering.
    /// The provided implementations reject stream capture and return after successful stream completion.
    /// Temporary device scratch or descriptor allocations remain alive through native use.
    ///
    /// # Errors
    ///
    /// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
    /// Scratch-size queries, device allocations or descriptor uploads, capture checks, and stream synchronization can also fail.
    /// Source and destination image sizes must match.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Inputs must be initialized and ready for native execution.
    /// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
    /// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
    /// NPP receives mutable source pointers; provide exclusive access to source storage during native use.
    /// Retain all allocations until native work completes.
    /// Native UF destinations must use linear CUDA allocations, rather than pitched allocations, with a byte stride of the source image width times `size_of::<u32>()`. The wrapper does not enforce this allocation or stride requirement.
    unsafe fn dispatch_to(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, Destination, C1>,
        norm: ImageNormalization,
    ) -> Result<()>;
}

unsafe impl<T> LabelMarkersUfToC1<u32> for T
where
    T: LabelMarkersUfC1,
{
    unsafe fn dispatch_to(
        stream_context: &StreamContext,
        source: &ImageView<'_, Self, C1>,
        destination: &mut ImageOutputView<'_, u32, C1>,
        norm: ImageNormalization,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { T::dispatch(stream_context, source, destination, norm) }
    }
}

/// Labels connected regions with the native union-find algorithm.
///
/// A connected region contains pixels with the same value. `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity.
/// Native marker IDs may have gaps and do not have a defined ordering.
/// The provided implementations reject stream capture and return after successful stream completion.
/// Temporary device scratch or descriptor allocations remain alive through native use.
///
/// # Errors
///
/// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
/// Scratch-size queries, device allocations or descriptor uploads, capture checks, and stream synchronization can also fail.
/// Source and destination image sizes must match.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Inputs must be initialized and ready for native execution.
/// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
/// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
/// NPP receives mutable source pointers; provide exclusive access to source storage during native use.
/// Retain all allocations until native work completes.
/// Native UF destinations must use linear CUDA allocations, rather than pitched allocations, with a byte stride of the source image width times `size_of::<u32>()`. The wrapper does not enforce this allocation or stride requirement.
pub unsafe fn label_markers_uf_to<T, Destination>(
    stream_context: &StreamContext,
    source: &ImageView<'_, T, C1>,
    destination: &mut ImageOutputView<'_, Destination, C1>,
    norm: ImageNormalization,
) -> Result<()>
where
    T: LabelMarkersUfToC1<Destination>,
    Destination: DataTypeLike,
{
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::dispatch_to(stream_context, source, destination, norm) }
}

/// Dispatches a batch of union-find connected-component label images over a common ROI.
///
/// # Safety
///
/// Implementations must preserve native element representations, connectivity, marker output semantics, and the documented execution contract.
/// Scratch-size queries must not underestimate native workspace requirements. Retain temporary resources through native completion.
pub unsafe trait LabelMarkersUfBatchC1: DataTypeLike + Sized {
    /// Labels connected regions with the native union-find algorithm.
    ///
    /// A connected region contains pixels with the same value. `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity.
    /// Native marker IDs may have gaps and do not have a defined ordering.
    /// The provided implementations reject stream capture and return after successful stream completion.
    /// Temporary device scratch or descriptor allocations remain alive through native use.
    /// The common ROI uses the smallest source or destination width and height in the batch. Pixels outside that ROI remain unwritten.
    ///
    /// # Errors
    ///
    /// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
    /// Scratch-size queries, device allocations or descriptor uploads, capture checks, and stream synchronization can also fail.
    /// Source and destination batches must have equal lengths of at least two, representable as `i32`. Each source and destination pair must have equal sizes.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Inputs must be initialized and ready for native execution.
    /// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
    /// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
    /// NPP receives mutable source pointers; provide exclusive access to source storage during native use.
    /// Retain all allocations until native work completes.
    /// Native UF destinations must use linear CUDA allocations, rather than pitched allocations, with a byte stride of the common ROI width times `size_of::<u32>()`. The wrapper does not enforce this allocation or stride requirement.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageOutputView<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()>;
}

unsafe impl LabelMarkersUfBatchC1 for u8 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageOutputView<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { label_markers_uf_batch_u8_to_u32_c1(stream_context, sources, destinations, norm) }
    }
}

unsafe impl LabelMarkersUfBatchC1 for u16 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageOutputView<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { label_markers_uf_batch_u16_to_u32_c1(stream_context, sources, destinations, norm) }
    }
}

unsafe impl LabelMarkersUfBatchC1 for u32 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageOutputView<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { label_markers_uf_batch_u32_c1(stream_context, sources, destinations, norm) }
    }
}

/// Labels connected regions with the native union-find algorithm.
///
/// A connected region contains pixels with the same value. `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity.
/// Native marker IDs may have gaps and do not have a defined ordering.
/// The provided implementations reject stream capture and return after successful stream completion.
/// Temporary device scratch or descriptor allocations remain alive through native use.
/// The common ROI uses the smallest source or destination width and height in the batch. Pixels outside that ROI remain unwritten.
///
/// # Errors
///
/// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
/// Scratch-size queries, device allocations or descriptor uploads, capture checks, and stream synchronization can also fail.
/// Source and destination batches must have equal lengths of at least two, representable as `i32`. Each source and destination pair must have equal sizes.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Inputs must be initialized and ready for native execution.
/// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
/// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
/// NPP receives mutable source pointers; provide exclusive access to source storage during native use.
/// Retain all allocations until native work completes.
/// Native UF destinations must use linear CUDA allocations, rather than pitched allocations, with a byte stride of the common ROI width times `size_of::<u32>()`. The wrapper does not enforce this allocation or stride requirement.
pub unsafe fn label_markers_uf_batch<T: LabelMarkersUfBatchC1>(
    stream_context: &StreamContext,
    sources: &[ImageView<'_, T, C1>],
    destinations: &mut [ImageOutputView<'_, u32, C1>],
    norm: ImageNormalization,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::dispatch(stream_context, sources, destinations, norm) }
}

/// Dispatches a batch of union-find connected-component label images with individual image sizes.
///
/// # Safety
///
/// Implementations must preserve native element representations, connectivity, marker output semantics, and the documented execution contract.
/// Scratch-size queries must not underestimate native workspace requirements. Retain temporary resources through native completion.
pub unsafe trait LabelMarkersUfBatchAdvancedC1: DataTypeLike + Sized {
    /// Labels connected regions with the native union-find algorithm.
    ///
    /// A connected region contains pixels with the same value. `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity.
    /// Native marker IDs may have gaps and do not have a defined ordering.
    /// The provided implementations reject stream capture and return after successful stream completion.
    /// Temporary device scratch or descriptor allocations remain alive through native use.
    /// Each pair uses its own image size. The largest destination width and height determine the native launch bounds.
    ///
    /// # Errors
    ///
    /// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
    /// Scratch-size queries, device allocations or descriptor uploads, capture checks, and stream synchronization can also fail.
    /// Source and destination batches must have equal lengths of at least two, representable as `i32`. Each source and destination pair must have equal sizes.
    ///
    /// # Safety
    ///
    /// Follow the [raw image execution contract](crate::image::raw).
    /// Inputs must be initialized and ready for native execution.
    /// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
    /// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
    /// NPP receives mutable source pointers; provide exclusive access to source storage during native use.
    /// Retain all allocations until native work completes.
    /// Native UF destinations must use linear CUDA allocations, rather than pitched allocations, with a byte stride of each destination image width times `size_of::<u32>()`. The wrapper does not enforce this allocation or stride requirement.
    unsafe fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageOutputView<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()>;
}

unsafe impl LabelMarkersUfBatchAdvancedC1 for u8 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageOutputView<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            label_markers_uf_batch_u8_to_u32_c1_advanced(
                stream_context,
                sources,
                destinations,
                norm,
            )
        }
    }
}

unsafe impl LabelMarkersUfBatchAdvancedC1 for u16 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageOutputView<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            label_markers_uf_batch_u16_to_u32_c1_advanced(
                stream_context,
                sources,
                destinations,
                norm,
            )
        }
    }
}

unsafe impl LabelMarkersUfBatchAdvancedC1 for u32 {
    unsafe fn dispatch(
        stream_context: &StreamContext,
        sources: &[ImageView<'_, Self, C1>],
        destinations: &mut [ImageOutputView<'_, u32, C1>],
        norm: ImageNormalization,
    ) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            label_markers_uf_batch_u32_c1_advanced(stream_context, sources, destinations, norm)
        }
    }
}

/// Labels connected regions with the native union-find algorithm.
///
/// A connected region contains pixels with the same value. `ImageNormalization::L1` selects four-way connectivity and `ImageNormalization::Infinity` selects eight-way connectivity.
/// Native marker IDs may have gaps and do not have a defined ordering.
/// The provided implementations reject stream capture and return after successful stream completion.
/// Temporary device scratch or descriptor allocations remain alive through native use.
/// Each pair uses its own image size. The largest destination width and height determine the native launch bounds.
///
/// # Errors
///
/// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
/// Scratch-size queries, device allocations or descriptor uploads, capture checks, and stream synchronization can also fail.
/// Source and destination batches must have equal lengths of at least two, representable as `i32`. Each source and destination pair must have equal sizes.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Inputs must be initialized and ready for native execution.
/// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
/// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
/// NPP receives mutable source pointers; provide exclusive access to source storage during native use.
/// Retain all allocations until native work completes.
/// Native UF destinations must use linear CUDA allocations, rather than pitched allocations, with a byte stride of each destination image width times `size_of::<u32>()`. The wrapper does not enforce this allocation or stride requirement.
pub unsafe fn label_markers_uf_batch_advanced<T: LabelMarkersUfBatchAdvancedC1>(
    stream_context: &StreamContext,
    sources: &[ImageView<'_, T, C1>],
    destinations: &mut [ImageOutputView<'_, u32, C1>],
    norm: ImageNormalization,
) -> Result<()> {
    // SAFETY: the caller upholds this function's safety contract.
    unsafe { T::dispatch(stream_context, sources, destinations, norm) }
}

/// Renumbers UF marker IDs in place to remove numbering gaps.
///
/// The device `new_max_label_ids` output receives one new maximum label ID per image.
/// The provided implementations reject stream capture and return after successful stream completion.
/// Temporary device scratch or descriptor allocations remain alive through native use.
/// All images use one common size and per-image scratch capacity.
///
/// # Errors
///
/// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
/// The batch must contain at least two images and fit in `i32`. Each image stride must equal its width times `size_of::<u32>()`.
/// `new_max_label_ids` must contain exactly one element per image. ROI-area arithmetic, scratch queries and allocations, descriptor uploads, capture checks, and synchronization can also fail.
/// All image sizes must match.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Inputs must be initialized and ready for native execution.
/// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
/// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
/// Each image must contain output from a matching UF label-generation call, with the same base pointer and ROI. Use linear CUDA allocations rather than pitched allocations.
/// Retain images and the count output until native work completes.
pub unsafe fn compress_marker_labels_uf_batch_u32_c1_in_place(
    stream_context: &StreamContext,
    source_destinations: &mut [ImageViewMut<'_, u32, C1>],
    new_max_label_ids: &mut (impl DeviceOutput<u32> + ?Sized),
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(new_max_label_ids)?;
    for image in source_destinations.iter() {
        stream_context.validate_region(image)?;
    }
    stream_context.stream().ensure_not_capturing()?;

    let (batch_size, roi, descriptors) = compress_marker_labels_uf_batch_descriptors(
        stream_context.cuda_context(),
        source_destinations,
    )?;
    validate_batch_label_output(new_max_label_ids, batch_size as usize)?;
    let starting_number = roi_area(roi)?;
    let per_image_bytes = compress_marker_labels_uf_buffer_size_u32_c1(starting_number)?;
    let (_buffers, mut buffer_descriptors) = compress_marker_labels_uf_batch_buffers(
        stream_context.cuda_context(),
        batch_size as usize,
        per_image_bytes,
    )?;
    let per_image_bytes =
        i32::try_from(per_image_bytes).map_err(|_| Error::out_of_range("buffer size"))?;

    // SAFETY: `nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    stream_context.sync_scope(|| unsafe {
        try_ffi!(sys::nppiCompressMarkerLabelsUFBatch_32u_C1IR_Ctx(
            descriptors.as_ptr().cast_mut().cast(),
            buffer_descriptors.as_mut_ptr().cast(),
            new_max_label_ids.as_mut_ptr().cast(),
            batch_size,
            roi.into(),
            per_image_bytes,
            stream_context.as_raw(),
        ))
    })?;

    Ok(())
}

/// Renumbers UF marker IDs in place to remove numbering gaps.
///
/// The device `new_max_label_ids` output receives one new maximum label ID per image.
/// The provided implementations reject stream capture and return after successful stream completion.
/// Temporary device scratch or descriptor allocations remain alive through native use.
/// Uses each image's own dimensions and scratch size. The largest width, height, and per-image scratch size provide native batch bounds.
///
/// # Errors
///
/// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
/// The batch must contain at least two images and fit in `i32`. Each image stride must equal its width times `size_of::<u32>()`.
/// `new_max_label_ids` must contain exactly one element per image. ROI-area arithmetic, scratch queries and allocations, descriptor uploads, capture checks, and synchronization can also fail.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Inputs must be initialized and ready for native execution.
/// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
/// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
/// Each image must contain output from a matching UF label-generation call, with the same base pointer and ROI. Use linear CUDA allocations rather than pitched allocations.
/// Retain images and the count output until native work completes.
pub unsafe fn compress_marker_labels_uf_batch_u32_c1_in_place_advanced(
    stream_context: &StreamContext,
    source_destinations: &mut [ImageViewMut<'_, u32, C1>],
    new_max_label_ids: &mut (impl DeviceOutput<u32> + ?Sized),
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(new_max_label_ids)?;
    for image in source_destinations.iter() {
        stream_context.validate_region(image)?;
    }
    stream_context.stream().ensure_not_capturing()?;

    let (batch_size, max_roi, descriptors) = compress_marker_labels_uf_batch_advanced_descriptors(
        stream_context.cuda_context(),
        source_destinations,
    )?;
    validate_batch_label_output(new_max_label_ids, batch_size as usize)?;
    let (largest_per_image_bytes, _buffers, mut buffer_descriptors) =
        compress_marker_labels_uf_batch_advanced_buffers(
            stream_context.cuda_context(),
            source_destinations,
        )?;
    let largest_per_image_bytes =
        i32::try_from(largest_per_image_bytes).map_err(|_| Error::out_of_range("buffer size"))?;

    // SAFETY: `nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    stream_context.sync_scope(|| unsafe {
        try_ffi!(sys::nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(
            descriptors.as_ptr().cast_mut().cast(),
            buffer_descriptors.as_mut_ptr().cast(),
            new_max_label_ids.as_mut_ptr().cast(),
            batch_size,
            max_roi.into(),
            largest_per_image_bytes,
            stream_context.as_raw(),
        ))
    })?;

    Ok(())
}

/// Returns the scratch capacity in bytes for compressing UF marker labels.
///
/// `starting_number` is the ROI width times height used during marker generation.
///
/// # Errors
///
/// Returns an error if NPP rejects the query or the returned byte count cannot be represented as `usize`.
pub fn compress_marker_labels_uf_buffer_size_u32_c1(starting_number: i32) -> Result<usize> {
    let mut bytes = 0;
    // SAFETY: `nppiCompressMarkerLabelsGetBufferSize_32u_C1R` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiCompressMarkerLabelsGetBufferSize_32u_C1R(
            starting_number,
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

/// Returns the marker-info list capacity in bytes for `max_marker_label_id`.
///
/// Use the maximum label ID returned by the corresponding UF label-compression call.
///
/// # Errors
///
/// Returns an error if NPP rejects the query or the returned byte count cannot be represented as `usize`.
pub fn compressed_marker_labels_uf_info_list_size_u32_c1(
    max_marker_label_id: u32,
) -> Result<usize> {
    let mut bytes = 0_u32;
    // SAFETY: `nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiCompressedMarkerLabelsUFGetInfoListSize_32u_C1R(
            max_marker_label_id,
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

/// Returns the block-segment list capacity in bytes for contour geometry generation.
///
/// The host counts, total contour pixel count, label count, and requested contour IDs must come from the matching UF contour-info results.
///
/// # Errors
///
/// Returns an error if `compressed_label_count + 1` overflows, the host count list has fewer than that many entries, or the contour IDs do not satisfy `first < last <= compressed_label_count`.
/// NPP query errors and returned byte-count conversion errors can also occur.
pub fn compressed_marker_labels_uf_contours_block_segment_list_size_c1(
    contours_pixel_counts_list_host: &mut [u32],
    total_image_pixel_contour_count: u32,
    compressed_label_count: u32,
    first_contour_geometry_list_id: u32,
    last_contour_geometry_list_id: u32,
) -> Result<usize> {
    validate_u32_list_host_output(
        contours_pixel_counts_list_host,
        compressed_label_count
            .checked_add(1)
            .ok_or_else(|| Error::out_of_range("contour count slots"))?,
        "contours pixel counts list",
    )?;
    validate_geometry_list_id_range(
        first_contour_geometry_list_id,
        last_contour_geometry_list_id,
        compressed_label_count,
    )?;
    let mut bytes = 0_u32;
    // SAFETY: `nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R` receives valid output locations.
    unsafe {
        try_ffi!(
            sys::nppiCompressedMarkerLabelsUFGetContoursBlockSegmentListSize_C1R(
                contours_pixel_counts_list_host.as_mut_ptr().cast(),
                total_image_pixel_contour_count,
                compressed_label_count,
                first_contour_geometry_list_id,
                last_contour_geometry_list_id,
                &raw mut bytes,
            )
        )?;
    }
    to_usize(bytes, "buffer size")
}

/// Returns the contour-geometry list capacity in bytes for the supplied pixel count.
///
/// `max_contour_pixel_geometry_info_count` is the final contour starting-offset sentinel produced by the matching contour-info call.
///
/// # Errors
///
/// Returns an error if NPP rejects the query or the returned byte count cannot be represented as `usize`.
pub fn compressed_marker_labels_uf_geometry_lists_size_c1(
    max_contour_pixel_geometry_info_count: u32,
) -> Result<usize> {
    let mut bytes = 0_u32;
    // SAFETY: `nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R` receives valid output locations.
    unsafe {
        try_ffi!(sys::nppiCompressedMarkerLabelsUFGetGeometryListsSize_C1R(
            max_contour_pixel_geometry_info_count,
            &raw mut bytes,
        ))?;
    }
    to_usize(bytes, "buffer size")
}

/// Extracts compressed-label statistics, contour images, directions, counts, and starting offsets.
///
/// All host outputs correspond to the same UF compressed-label image and `max_marker_label_id`.
/// Submits native work without an explicit wrapper synchronization.
/// Host and device buffers remain borrowed by native work until completion; returning successfully does not release those lifetime obligations.
///
/// # Errors
///
/// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
/// Contour and direction images must match the label-image size. The label-image stride must equal its width times `size_of::<u32>()`.
/// The marker-info allocation must cover the queried byte count. Both count lists require at least `max_marker_label_id` entries; both offset lists require at least `max_marker_label_id + 1`, with checked sentinel arithmetic.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Inputs must be initialized and ready for native execution.
/// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
/// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
/// The label image must be a linear CUDA allocation produced by the UF generation and compression sequence, using its original base pointer and ROI. NPP receives a mutable label-image pointer; provide exclusive access during native use.
/// Keep host outputs and device buffers valid and exclusively accessible through queued completion, including submission errors and every captured replay. Host allocations must meet native host-memory requirements.
pub unsafe fn compressed_marker_labels_uf_info_u32_c1(
    stream_context: &StreamContext,
    compressed_marker_labels: &ImageView<'_, u32, C1>,
    max_marker_label_id: u32,
    marker_labels_info_list: &mut DeviceMemory<CompressedMarkerLabelsInfo>,
    contours_image: &mut ImageViewMut<'_, u8, C1>,
    contours_direction_image: &mut ImageViewMut<'_, ContourPixelDirectionInfo, C1>,
    contours_totals_info_host: &mut ContourTotalsInfo,
    contours_pixel_counts_list_dev: &mut DeviceMemory<u32>,
    contours_pixel_counts_list_host: &mut [u32],
    contours_pixel_starting_offset_dev: &mut DeviceMemory<u32>,
    contours_pixel_starting_offset_host: &mut [u32],
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(compressed_marker_labels)?;
    stream_context.validate_region(marker_labels_info_list)?;
    stream_context.validate_region(contours_image)?;
    stream_context.validate_region(contours_direction_image)?;
    stream_context.validate_region(contours_pixel_counts_list_dev)?;
    stream_context.validate_region(contours_pixel_starting_offset_dev)?;

    validate_same_size(compressed_marker_labels.size(), contours_image.size())?;
    validate_same_size(
        compressed_marker_labels.size(),
        contours_direction_image.size(),
    )?;
    validate_u32_label_marker_step(
        compressed_marker_labels.size(),
        compressed_marker_labels.step(),
    )?;
    validate_marker_label_info_output(marker_labels_info_list, max_marker_label_id)?;
    validate_u32_list_output(
        contours_pixel_counts_list_dev,
        contours_pixel_counts_list_host,
        max_marker_label_id,
        "contours pixel counts list",
    )?;
    validate_u32_list_output(
        contours_pixel_starting_offset_dev,
        contours_pixel_starting_offset_host,
        max_marker_label_id
            .checked_add(1)
            .ok_or_else(|| Error::out_of_range("contour offset sentinel"))?,
        "contours pixel starting offset list",
    )?;

    // SAFETY: `nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
    unsafe {
        try_ffi!(sys::nppiCompressedMarkerLabelsUFInfo_32u_C1R_Ctx(
            compressed_marker_labels.as_ptr().cast_mut().cast(),
            compressed_marker_labels.step(),
            compressed_marker_labels.size().into(),
            max_marker_label_id,
            marker_labels_info_list.as_mut_ptr().cast(),
            contours_image.as_mut_ptr().cast(),
            contours_image.step(),
            contours_direction_image.as_mut_ptr().cast(),
            contours_direction_image.step(),
            ptr::from_mut(contours_totals_info_host).cast(),
            contours_pixel_counts_list_dev.as_mut_ptr().cast(),
            contours_pixel_counts_list_host.as_mut_ptr().cast(),
            contours_pixel_starting_offset_dev.as_mut_ptr().cast(),
            contours_pixel_starting_offset_host.as_mut_ptr().cast(),
            stream_context.as_raw(),
        ))?;
    }
    Ok(())
}

/// Generates ordered contour geometry from matching UF contour-info results.
///
/// `output_in_counterclockwise_order` selects counterclockwise ordering when true and clockwise ordering otherwise. Counts, offsets, label information, and direction images must describe the same image and ROI.
/// The native routine may skip contours larger than 256K pixels unless the requested range selects that contour alone. Only native-reported contour geometry is valid.
/// Submits native work without an explicit wrapper synchronization.
/// Host and device buffers remain borrowed by native work until completion; returning successfully does not release those lifetime obligations.
///
/// # Errors
///
/// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
/// Both marker-info lists must cover their queried byte count. Device and host geometry lists must have equal lengths and cover the query for the final host offset sentinel.
/// The host offset list must contain the sentinel at `max_marker_label_id`; both offset lists require at least `max_marker_label_id + 1` entries. The device count list and both found-count lists require at least `max_marker_label_id` entries.
/// Contour IDs must satisfy `first < last <= max_marker_label_id`. The host geometry-image byte span must fit its supplied step and image size without arithmetic overflow. Device and host block-segment list lengths must match.
///
/// # Safety
///
/// Follow the [raw image execution contract](crate::image::raw).
/// Inputs must be initialized and ready for native execution.
/// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
/// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
/// All metadata, counts, offsets, directions, and total pixel counts must come from the matching UF contour workflow. Host offsets read during validation must already be ready.
/// Size both block-segment lists using the matching block-segment size query; this wrapper checks their lengths agree but does not prove sufficient capacity. The host image stride must satisfy native image layout requirements.
/// Keep all host and device allocations valid and exclusively accessible through queued completion, including submission errors and captured replay. Host allocations must meet native host-memory requirements.
pub unsafe fn compressed_marker_labels_uf_contours_generate_geometry_lists_c1(
    stream_context: &StreamContext,
    marker_labels_info_list_dev: &mut DeviceMemory<CompressedMarkerLabelsInfo>,
    marker_labels_info_list_host: &mut [CompressedMarkerLabelsInfo],
    contours_direction_image_dev: &mut ImageViewMut<'_, ContourPixelDirectionInfo, C1>,
    contours_pixel_geometry_lists_dev: &mut DeviceMemory<ContourPixelGeometryInfo>,
    contours_pixel_geometry_lists_host: &mut [ContourPixelGeometryInfo],
    contours_geometry_image_host: &mut [u8],
    contours_geometry_image_step: i32,
    contours_pixel_counts_list_dev: &mut DeviceMemory<u32>,
    contours_pixels_found_list_dev: &mut DeviceMemory<u32>,
    contours_pixels_found_list_host: &mut [u32],
    contours_pixels_starting_offset_dev: &mut DeviceMemory<u32>,
    contours_pixels_starting_offset_host: &mut [u32],
    total_image_pixel_contour_count: u32,
    max_marker_label_id: u32,
    first_contour_geometry_list_id: u32,
    last_contour_geometry_list_id: u32,
    contours_block_segment_list_dev: &mut DeviceMemory<ContourBlockSegment>,
    contours_block_segment_list_host: &mut [ContourBlockSegment],
    output_in_counterclockwise_order: bool,
) -> Result<()> {
    stream_context.cuda_context().bind()?;
    stream_context.validate_region(marker_labels_info_list_dev)?;
    stream_context.validate_region(contours_direction_image_dev)?;
    stream_context.validate_region(contours_pixel_geometry_lists_dev)?;
    stream_context.validate_region(contours_pixel_counts_list_dev)?;
    stream_context.validate_region(contours_pixels_found_list_dev)?;
    stream_context.validate_region(contours_pixels_starting_offset_dev)?;
    stream_context.validate_region(contours_block_segment_list_dev)?;

    validate_marker_label_info_host_output(marker_labels_info_list_host, max_marker_label_id)?;
    validate_marker_label_info_output(marker_labels_info_list_dev, max_marker_label_id)?;
    validate_u32_list_host_output(
        contours_pixels_starting_offset_host,
        max_marker_label_id
            .checked_add(1)
            .ok_or_else(|| Error::out_of_range("contour offset count"))?,
        "contour offsets",
    )?;
    validate_geometry_list_buffers(
        contours_pixel_geometry_lists_dev,
        contours_pixel_geometry_lists_host,
        contours_pixels_starting_offset_host[max_marker_label_id as usize],
    )?;
    validate_u32_list_dev_output(
        contours_pixel_counts_list_dev,
        max_marker_label_id,
        "contours pixel counts list",
    )?;
    validate_u32_list_output(
        contours_pixels_found_list_dev,
        contours_pixels_found_list_host,
        max_marker_label_id,
        "contours pixels found list",
    )?;
    validate_u32_list_output(
        contours_pixels_starting_offset_dev,
        contours_pixels_starting_offset_host,
        max_marker_label_id
            .checked_add(1)
            .ok_or_else(|| Error::out_of_range("contour offset sentinel"))?,
        "contours pixel starting offset list",
    )?;
    validate_geometry_list_id_range(
        first_contour_geometry_list_id,
        last_contour_geometry_list_id,
        max_marker_label_id,
    )?;
    validate_contours_geometry_image_host(
        contours_geometry_image_host,
        contours_direction_image_dev.size(),
        contours_geometry_image_step,
    )?;
    // Native segment capacity depends on matching host pixel counts, not on
    // the initially empty found-count output. The raw caller supplies query-sized storage.
    validate_block_segment_inputs(
        contours_block_segment_list_dev,
        contours_block_segment_list_host,
    )?;

    // SAFETY: `nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx` receives live handles owned by the wrappers and valid output locations.
    unsafe {
        try_ffi!(
            sys::nppiCompressedMarkerLabelsUFContoursGenerateGeometryLists_C1R_Ctx(
                marker_labels_info_list_dev.as_mut_ptr().cast(),
                marker_labels_info_list_host.as_mut_ptr().cast(),
                contours_direction_image_dev.as_mut_ptr().cast(),
                contours_direction_image_dev.step(),
                contours_pixel_geometry_lists_dev.as_mut_ptr().cast(),
                contours_pixel_geometry_lists_host.as_mut_ptr().cast(),
                contours_geometry_image_host.as_mut_ptr().cast(),
                contours_geometry_image_step,
                contours_pixel_counts_list_dev.as_mut_ptr().cast(),
                contours_pixels_found_list_dev.as_mut_ptr().cast(),
                contours_pixels_found_list_host.as_mut_ptr().cast(),
                contours_pixels_starting_offset_dev.as_mut_ptr().cast(),
                contours_pixels_starting_offset_host.as_mut_ptr().cast(),
                total_image_pixel_contour_count,
                max_marker_label_id,
                first_contour_geometry_list_id,
                last_contour_geometry_list_id,
                contours_block_segment_list_dev.as_mut_ptr().cast(),
                contours_block_segment_list_host.as_mut_ptr().cast(),
                u32::from(output_in_counterclockwise_order),
                contours_direction_image_dev.size().into(),
                stream_context.as_raw(),
            )
        )?;
    }
    Ok(())
}

macro_rules! impl_contours_image_marching_squares_interpolation {
    ($name:ident, $point_ty:ty, $ffi_name:ident) => {
        /// Interpolates integer contour coordinates with native marching squares into floating-point contour coordinates.
        ///
        /// The contour images, directions, geometry, found counts, and offsets must come from the matching UF contour workflow. Only native-reported contour results are valid.
        /// Native processing may skip contours larger than 256K pixels unless the requested range selects that contour alone.
        /// Submits native work without an explicit wrapper synchronization.
        /// Host and device buffers remain borrowed by native work until completion; returning successfully does not release those lifetime obligations.
        ///
        /// # Errors
        ///
        /// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
        /// All three image sizes must match. Contour IDs must satisfy `first < last <= max_marker_label_id`.
        /// The host offset sentinel at `max_marker_label_id` must exist. Both geometry lists must have equal lengths and cover the geometry-size query for that sentinel.
        /// The host found-count list and both starting-offset lists require at least `max_marker_label_id` entries, and the host offset list additionally needs its sentinel. The interpolated device geometry list must have exactly `total_image_pixel_contour_count` elements. Device and host block-segment list lengths must match.
        ///
        /// # Safety
        ///
        /// Follow the [raw image execution contract](crate::image::raw).
        /// Inputs must be initialized and ready for native execution.
        /// Keep every native-writable allocation exclusively accessible and disjoint from all other allocations used by the call.
        /// Native layouts, strides, alignment, allocation bounds, and input contents must be valid.
        /// NPP receives mutable pointers to the contour source and metadata; provide exclusive access to all such storage during native use. The host offset sentinel must be ready before this call reads it.
        /// Use matching query-sized block-segment storage; equality of list lengths does not establish adequate capacity.
        /// Retain host and device allocations through queued completion, including submission errors and every captured replay. Host allocations must meet native host-memory requirements.
        pub unsafe fn $name(
            stream_context: &StreamContext,
            contours_image_dev: &ImageView<'_, u8, C1>,
            contours_interpolated_image_dev: &mut ImageViewMut<'_, $point_ty, C1>,
            contours_direction_image_dev: &mut ImageViewMut<'_, ContourPixelDirectionInfo, C1>,
            contours_pixel_geometry_lists_dev: &mut DeviceMemory<ContourPixelGeometryInfo>,
            contours_pixel_geometry_lists_host: &mut [ContourPixelGeometryInfo],
            contours_interpolated_geometry_lists_dev: &mut DeviceMemory<$point_ty>,
            contours_pixels_found_list_host: &mut [u32],
            contours_pixels_starting_offset_dev: &mut DeviceMemory<u32>,
            contours_pixels_starting_offset_host: &mut [u32],
            total_image_pixel_contour_count: u32,
            max_marker_label_id: u32,
            first_contour_geometry_list_id: u32,
            last_contour_geometry_list_id: u32,
            contours_block_segment_list_dev: &mut DeviceMemory<ContourBlockSegment>,
            contours_block_segment_list_host: &mut [ContourBlockSegment],
        ) -> Result<()> {
            stream_context.cuda_context().bind()?;
            stream_context.validate_region(contours_image_dev)?;
            stream_context.validate_region(contours_interpolated_image_dev)?;
            stream_context.validate_region(contours_direction_image_dev)?;
            stream_context.validate_region(contours_pixel_geometry_lists_dev)?;
            stream_context.validate_region(contours_interpolated_geometry_lists_dev)?;
            stream_context.validate_region(contours_pixels_starting_offset_dev)?;
            stream_context.validate_region(contours_block_segment_list_dev)?;

            validate_same_size(
                contours_image_dev.size(),
                contours_interpolated_image_dev.size(),
            )?;
            validate_same_size(
                contours_image_dev.size(),
                contours_direction_image_dev.size(),
            )?;
            validate_geometry_list_id_range(
                first_contour_geometry_list_id,
                last_contour_geometry_list_id,
                max_marker_label_id,
            )?;
            validate_geometry_list_buffers(
                contours_pixel_geometry_lists_dev,
                contours_pixel_geometry_lists_host,
                *contours_pixels_starting_offset_host
                    .get(max_marker_label_id as usize)
                    .ok_or_else(|| Error::out_of_range("contour offset sentinel"))?,
            )?;
            validate_u32_list_host_output(
                contours_pixels_found_list_host,
                max_marker_label_id,
                "contours pixels found list",
            )?;
            validate_u32_list_device_and_host_output(
                contours_pixels_starting_offset_dev,
                contours_pixels_starting_offset_host,
                max_marker_label_id,
                "contours pixel starting offset list",
            )?;
            validate_typed_output_len(
                contours_interpolated_geometry_lists_dev.len(),
                total_image_pixel_contour_count,
                "contours interpolated geometry lists",
            )?;
            validate_block_segment_inputs(
                contours_block_segment_list_dev,
                contours_block_segment_list_host,
            )?;

            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                try_ffi!(sys::$ffi_name(
                    contours_image_dev.as_ptr().cast_mut().cast(),
                    contours_image_dev.step(),
                    contours_interpolated_image_dev.as_mut_ptr().cast(),
                    contours_interpolated_image_dev.step(),
                    contours_direction_image_dev.as_mut_ptr().cast(),
                    contours_direction_image_dev.step(),
                    contours_pixel_geometry_lists_dev.as_mut_ptr().cast(),
                    contours_pixel_geometry_lists_host
                        .as_ptr()
                        .cast_mut()
                        .cast(),
                    contours_interpolated_geometry_lists_dev.as_mut_ptr().cast(),
                    contours_pixels_found_list_host.as_mut_ptr().cast(),
                    contours_pixels_starting_offset_dev
                        .as_ptr()
                        .cast_mut()
                        .cast(),
                    contours_pixels_starting_offset_host
                        .as_ptr()
                        .cast_mut()
                        .cast(),
                    total_image_pixel_contour_count,
                    max_marker_label_id,
                    first_contour_geometry_list_id,
                    last_contour_geometry_list_id,
                    contours_block_segment_list_dev.as_mut_ptr().cast(),
                    contours_block_segment_list_host.as_mut_ptr().cast(),
                    contours_image_dev.size().into(),
                    stream_context.as_raw(),
                ))?;
            }
            Ok(())
        }
    };
}

impl_contours_image_marching_squares_interpolation!(
    contours_image_marching_squares_interpolation_32f_c1,
    Point32f,
    nppiContoursImageMarchingSquaresInterpolation_32f_C1R_Ctx
);

impl_contours_image_marching_squares_interpolation!(
    contours_image_marching_squares_interpolation_64f_c1,
    Point64f,
    nppiContoursImageMarchingSquaresInterpolation_64f_C1R_Ctx
);
