use std::{mem::size_of, sync::Arc};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, view::DeviceView};

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        filtering,
        view::{C1, ImageView, ImageViewMut},
    },
    pipeline::image::ContiguousImage,
    types::{
        CompressedMarkerLabelsInfo, ContourBlockSegment, ContourPixelDirectionInfo,
        ContourPixelGeometryInfo, ContourTotalsInfo, Size,
    },
    utility::{checked_len, to_i32},
};

mod geometry;

pub use geometry::{ContourGeometry, ContourOrder};

mod interpolation;

pub use interpolation::InterpolatedContours;

/// Completed union-find labels in their original dense native allocation.
///
/// Read-only access preserves the pointer, extent, context, and values required by compression.
/// Raw writes must preserve the native labeling invariant; consume `into_image` for ordinary mutation.
#[derive(Debug)]
pub struct MarkerLabels {
    image: ContiguousImage<u32, C1>,
}

impl MarkerLabels {
    /// Adopts completed output from a public raw union-find labeling operation.
    ///
    /// # Safety
    /// The image must be unchanged, completed NPP union-find output in its original
    /// ordinary allocation, CUDA context, and exact ROI.
    #[must_use]
    pub unsafe fn from_image(image: ContiguousImage<u32, C1>) -> Self {
        Self { image }
    }

    /// Returns the CUDA context of this result's device allocations.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.image.context()
    }

    /// Returns the label-image width and height in pixels.
    #[must_use]
    pub fn size(&self) -> Size {
        self.image.size()
    }

    /// Borrows the completed label image without changing its native layout or values.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored allocation and image geometry cannot form a valid image view.
    pub fn labels(&self) -> Result<ImageView<'_, u32, C1>> {
        self.image.view()
    }

    /// Returns the backing device allocation for the label image.
    #[must_use]
    pub fn values(&self) -> &DeviceMemory<u32> {
        self.image.values()
    }

    /// Returns the owned dense label image for ordinary mutation.
    #[must_use]
    pub fn into_image(self) -> ContiguousImage<u32, C1> {
        self.image
    }

    /// Compresses the completed labels in their original allocation.
    ///
    /// Returns a completed compressed-label owner with the matching native maximum label ID. Retains the original allocation and exact image extent.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if the selected CUDA context differs from the image context, or if the pixel count overflows or exceeds `i32`.
    /// Image view construction, CUDA context binding, native scratch queries or allocation, NPP compression, and stream synchronization can also fail.
    pub fn compress(mut self, stream_context: &StreamContext) -> Result<CompressedMarkerLabels> {
        stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(self.image.values().allocation()))?;
        stream_context.stream().ensure_not_capturing()?;
        if self.context() != stream_context.cuda_context() {
            return Err(Error::StreamContextMismatch);
        }
        let starting_number = to_i32(checked_len(self.size(), 1)?, "label pixels")?;
        // SAFETY: ownership preserves the unchanged original native grid through completion.
        let max_label_id = unsafe {
            filtering::compress_marker_labels_uf_u32_c1_in_place(
                stream_context,
                &mut self.image.view_mut()?,
                starting_number,
            )?
        };
        Ok(CompressedMarkerLabels {
            image: self.image,
            max_label_id,
        })
    }

    /// Compresses original union-find grids, including grids with different dimensions.
    /// Every grid must share the selected actual CUDA context.
    ///
    /// Returns results in input order after native compression and count download complete. An empty input returns an empty vector; a single image uses single-image compression.
    ///
    /// # Errors
    ///
    /// Returns an error during capture or if any image belongs to another CUDA context.
    /// For batches of two or more images, batch-count, pixel-area, and scratch-size conversions must fit their native ranges.
    /// Image view creation, scratch or descriptor allocation, count allocation or download, NPP compression, and stream synchronization can also fail.
    pub fn compress_batch(
        stream_context: &StreamContext,
        mut labels: Vec<Self>,
    ) -> Result<Vec<CompressedMarkerLabels>> {
        for labels in &labels {
            stream_context
                .stream()
                .defer_drop_until_capture_end(Arc::clone(labels.image.values().allocation()))?;
        }
        stream_context.stream().ensure_not_capturing()?;
        for image in &labels {
            if image.context() != stream_context.cuda_context() {
                return Err(Error::StreamContextMismatch);
            }
        }
        if labels.len() < 2 {
            return labels
                .into_iter()
                .map(|image| image.compress(stream_context))
                .collect();
        }
        let mut counts =
            DeviceMemory::<u32>::create_uninit(stream_context.cuda_context(), labels.len())?;
        {
            let mut views = labels
                .iter_mut()
                .map(|labels| labels.image.view_mut())
                .collect::<Result<Vec<_>>>()?;
            // SAFETY: all original dense grids remain exclusively owned through the completed helper.
            unsafe {
                filtering::compress_marker_labels_uf_batch_u32_c1_in_place_advanced(
                    stream_context,
                    &mut views,
                    &mut counts,
                )?;
            }
        }
        // SAFETY: completed batch compression writes one count per owned grid.
        let counts = unsafe { counts.assume_init() }.copy_to_host_vec()?;
        Ok(labels
            .into_iter()
            .zip(counts)
            .map(|(labels, max_label_id)| CompressedMarkerLabels {
                image: labels.image,
                max_label_id,
            })
            .collect())
    }
}

/// Compressed labels retaining their original native allocation and matching maximum ID.
/// Raw writes must preserve the native compression invariant; consume `into_image` for ordinary mutation.
#[derive(Debug)]
pub struct CompressedMarkerLabels {
    image: ContiguousImage<u32, C1>,
    max_label_id: u32,
}

impl CompressedMarkerLabels {
    /// Adopts completed output from raw union-find compression.
    ///
    /// # Safety
    /// Preserve the original labeling allocation, context, ROI, and compressed values.
    /// `max_label_id` must be the corresponding native compression result.
    #[must_use]
    pub unsafe fn from_image(image: ContiguousImage<u32, C1>, max_label_id: u32) -> Self {
        Self {
            image,
            max_label_id,
        }
    }

    /// Returns the CUDA context of this result's device allocations.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.image.context()
    }

    /// Returns the label-image width and height in pixels.
    #[must_use]
    pub fn size(&self) -> Size {
        self.image.size()
    }

    /// Returns the exclusive upper bound of native label IDs.
    #[must_use]
    pub fn label_count(&self) -> u32 {
        self.max_label_id
    }

    /// Borrows the completed label image without changing its native layout or values.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored allocation and image geometry cannot form a valid image view.
    pub fn labels(&self) -> Result<ImageView<'_, u32, C1>> {
        self.image.view()
    }

    /// Returns the backing device allocation for the label image.
    #[must_use]
    pub fn values(&self) -> &DeviceMemory<u32> {
        self.image.values()
    }

    /// Returns the owned dense label image for ordinary mutation.
    #[must_use]
    pub fn into_image(self) -> ContiguousImage<u32, C1> {
        self.image
    }

    /// Returns the native contour-geometry list capacity in bytes for the final pixel-offset sentinel.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the query or the returned byte count cannot be represented as `usize`.
    pub fn geometry_lists_size(max_contour_pixel_geometry_info_count: u32) -> Result<usize> {
        filtering::compressed_marker_labels_uf_geometry_lists_size_c1(
            max_contour_pixel_geometry_info_count,
        )
    }

    /// Returns the native block-segment list capacity in bytes for matching host counts and contour IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if `compressed_label_count + 1` overflows, the host count list has fewer than that many entries, or the contour IDs do not satisfy `first < last <= compressed_label_count`.
    /// NPP query errors and returned byte-count conversion errors can also occur.
    pub fn contours_block_segment_list_size(
        contours_pixel_counts_list_host: &mut [u32],
        total_image_pixel_contour_count: u32,
        compressed_label_count: u32,
        first_contour_geometry_list_id: u32,
        last_contour_geometry_list_id: u32,
    ) -> Result<usize> {
        filtering::compressed_marker_labels_uf_contours_block_segment_list_size_c1(
            contours_pixel_counts_list_host,
            total_image_pixel_contour_count,
            compressed_label_count,
            first_contour_geometry_list_id,
            last_contour_geometry_list_id,
        )
    }

    /// Completes geometry generation with caller-owned native metadata.
    ///
    /// Prepares device metadata and output regions, waiting for their producers, and completes geometry generation before returning successfully.
    ///
    /// # Errors
    ///
    /// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
    /// Both marker-info lists must cover their queried byte count. Device and host geometry lists must have equal lengths and cover the query for the final host offset sentinel.
    /// The host offset list must contain the sentinel at `self.label_count()`; both offset lists require at least `self.label_count() + 1` entries. The device count list and both found-count lists require at least `self.label_count()` entries.
    /// Contour IDs must satisfy `first < last <= self.label_count()`. The host geometry-image byte span must fit its supplied step and image size without arithmetic overflow. Device and host block-segment list lengths must match.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// All images, metadata, counts, offsets, ranges, and totals must correspond to the same prior native contour-information call for these labels. Host offset sentinels read during validation must already be ready.
    /// Size both block-segment lists with the matching native query; matching lengths alone do not prove adequate capacity. The host image step must satisfy native row-layout requirements.
    /// Keep every native-writable host or device allocation exclusively accessible and disjoint from the other allocations until native use completes. Host buffers must meet native host-memory requirements.
    pub unsafe fn generate_contour_geometry_lists_into(
        &self,
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
        first_contour_geometry_list_id: u32,
        last_contour_geometry_list_id: u32,
        contours_block_segment_list_dev: &mut DeviceMemory<ContourBlockSegment>,
        contours_block_segment_list_host: &mut [ContourBlockSegment],
        output_in_counterclockwise_order: bool,
    ) -> Result<()> {
        stream_context.sync_scope(|| {
            stream_context.prepare_region(marker_labels_info_list_dev)?;
            stream_context.prepare_region(contours_direction_image_dev)?;
            stream_context.prepare_region(contours_pixel_geometry_lists_dev)?;
            stream_context.prepare_region(contours_pixel_counts_list_dev)?;
            stream_context.prepare_region(contours_pixels_found_list_dev)?;
            stream_context.prepare_region(contours_pixels_starting_offset_dev)?;
            stream_context.prepare_region(contours_block_segment_list_dev)?;
            // SAFETY: the completed scope retains every device and host borrow through native use.
            unsafe {
                filtering::compressed_marker_labels_uf_contours_generate_geometry_lists_c1(
                    stream_context,
                    marker_labels_info_list_dev,
                    marker_labels_info_list_host,
                    contours_direction_image_dev,
                    contours_pixel_geometry_lists_dev,
                    contours_pixel_geometry_lists_host,
                    contours_geometry_image_host,
                    contours_geometry_image_step,
                    contours_pixel_counts_list_dev,
                    contours_pixels_found_list_dev,
                    contours_pixels_found_list_host,
                    contours_pixels_starting_offset_dev,
                    contours_pixels_starting_offset_host,
                    total_image_pixel_contour_count,
                    self.max_label_id,
                    first_contour_geometry_list_id,
                    last_contour_geometry_list_id,
                    contours_block_segment_list_dev,
                    contours_block_segment_list_host,
                    output_in_counterclockwise_order,
                )
            }
        })
    }

    /// Returns the marker-info list capacity in bytes for this compressed label set.
    ///
    /// # Errors
    ///
    /// Returns an error if NPP rejects the query or the returned byte count cannot be represented as `usize`.
    pub fn info_list_size(&self) -> Result<usize> {
        filtering::compressed_marker_labels_uf_info_list_size_u32_c1(self.max_label_id)
    }

    /// Generates corresponding contour images and metadata from this native labeling result.
    /// Sparse updates start with initialized zero backgrounds and native-sized working arrays.
    ///
    /// Returns completed, mutually corresponding contour images, marker information, counts, offsets, and totals. The label allocation remains owned by the returned result.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if the selected context differs from the label-image context, or if checked image or list sizing fails.
    /// Native size queries, initialized device allocation, view construction, context binding, NPP contour extraction, and stream synchronization can also fail.
    pub fn contours(self, stream_context: &StreamContext) -> Result<Contours> {
        stream_context
            .stream()
            .defer_drop_until_capture_end(Arc::clone(self.image.values().allocation()))?;
        stream_context.stream().ensure_not_capturing()?;
        if self.context() != stream_context.cuda_context() {
            return Err(Error::StreamContextMismatch);
        }
        let size = self.size();
        let pixels = checked_len(size, 1)?;
        let slots = self
            .info_list_size()?
            .div_ceil(size_of::<CompressedMarkerLabelsInfo>());
        let context = self.context();
        let mut info = DeviceMemory::<CompressedMarkerLabelsInfo>::zeroed(context, slots)?;
        let mut contours =
            ContiguousImage::<u8, C1>::from_buffer(DeviceMemory::zeroed(context, pixels)?, size)?;
        let mut directions = ContiguousImage::<ContourPixelDirectionInfo, C1>::from_buffer(
            DeviceMemory::zeroed(context, pixels)?,
            size,
        )?;
        let mut counts = DeviceMemory::<u32>::zeroed(context, slots)?;
        let mut offsets = DeviceMemory::<u32>::zeroed(context, slots)?;
        let mut counts_host = vec![0; slots];
        let mut offsets_host = vec![0; slots];
        let mut totals = ContourTotalsInfo::default();
        stream_context.sync_scope(|| {
            // SAFETY: this owner preserves the labeling protocol and retains initialized
            // sparse-update backgrounds and every corresponding metadata buffer through completion.
            unsafe {
                filtering::compressed_marker_labels_uf_info_u32_c1(
                    stream_context,
                    &self.labels()?,
                    self.max_label_id,
                    &mut info,
                    &mut contours.view_mut()?,
                    &mut directions.view_mut()?,
                    &mut totals,
                    &mut counts,
                    &mut counts_host,
                    &mut offsets,
                    &mut offsets_host,
                )
            }
        })?;
        Ok(Contours {
            labels: self,
            info,
            pixel_image: contours,
            directions,
            totals,
            counts,
            counts_host,
            offsets,
            offsets_host,
        })
    }

    /// Completes native contour information generation into caller-managed storage.
    ///
    /// Prepares the label image and device outputs, waiting for producers, and returns native totals after successful stream completion.
    ///
    /// # Errors
    ///
    /// CUDA context binding, image or buffer context validation, NPP execution, and checked count or byte-size conversions can fail.
    /// Contour and direction images must match the label-image size. The label-image stride must equal its width times `size_of::<u32>()`.
    /// The marker-info allocation must cover the queried byte count. Both count lists require at least `self.label_count()` entries; both offset lists require at least `self.label_count() + 1`, with checked sentinel arithmetic.
    /// Capture checks, preparing regions, waiting for producers, and stream synchronization can also fail.
    ///
    /// # Safety
    ///
    /// Initialize all sparse-update backgrounds and metadata required by NPP. Outputs must belong to this call and must not alias one another or the label image.
    /// NPP receives a mutable label-image pointer; ensure exclusive native access to that allocation until this call returns.
    /// Host outputs must meet native host-memory requirements and remain exclusively accessible until native work completes.
    pub unsafe fn info_into(
        &self,
        stream_context: &StreamContext,
        marker_labels_info_list: &mut DeviceMemory<CompressedMarkerLabelsInfo>,
        contours_image: &mut ImageViewMut<'_, u8, C1>,
        contours_direction_image: &mut ImageViewMut<'_, ContourPixelDirectionInfo, C1>,
        contours_pixel_counts_list_dev: &mut DeviceMemory<u32>,
        contours_pixel_counts_list_host: &mut [u32],
        contours_pixel_starting_offset_dev: &mut DeviceMemory<u32>,
        contours_pixel_starting_offset_host: &mut [u32],
    ) -> Result<ContourTotalsInfo> {
        stream_context.stream().ensure_not_capturing()?;
        let source = self.labels()?;
        stream_context.prepare_region(&source)?;
        stream_context.prepare_region(marker_labels_info_list)?;
        stream_context.prepare_region(contours_image)?;
        stream_context.prepare_region(contours_direction_image)?;
        stream_context.prepare_region(contours_pixel_counts_list_dev)?;
        stream_context.prepare_region(contours_pixel_starting_offset_dev)?;
        let mut totals = ContourTotalsInfo::default();
        // SAFETY: the caller upholds this function's safety contract.
        stream_context.sync_scope(|| unsafe {
            filtering::compressed_marker_labels_uf_info_u32_c1(
                stream_context,
                &source,
                self.max_label_id,
                marker_labels_info_list,
                contours_image,
                contours_direction_image,
                &mut totals,
                contours_pixel_counts_list_dev,
                contours_pixel_counts_list_host,
                contours_pixel_starting_offset_dev,
                contours_pixel_starting_offset_host,
            )
        })?;
        Ok(totals)
    }
}

/// Corresponding images, counts, offsets, and region information from one completed native call.
/// Raw mutations must preserve the association between this call's images and metadata.
#[derive(Debug)]
pub struct Contours {
    labels: CompressedMarkerLabels,
    info: DeviceMemory<CompressedMarkerLabelsInfo>,
    pixel_image: ContiguousImage<u8, C1>,
    directions: ContiguousImage<ContourPixelDirectionInfo, C1>,
    totals: ContourTotalsInfo,
    counts: DeviceMemory<u32>,
    counts_host: Vec<u32>,
    offsets: DeviceMemory<u32>,
    offsets_host: Vec<u32>,
}

impl Contours {
    fn ensure_eager(&self, stream_context: &StreamContext) -> Result<()> {
        for owner in [
            self.labels.values().allocation(),
            self.info.allocation(),
            self.pixel_image.values().allocation(),
            self.directions.values().allocation(),
            self.counts.allocation(),
            self.offsets.allocation(),
        ] {
            stream_context
                .stream()
                .defer_drop_until_capture_end(Arc::clone(owner))?;
        }
        stream_context.stream().ensure_not_capturing()?;
        if self.context() != stream_context.cuda_context() {
            return Err(Error::StreamContextMismatch);
        }
        Ok(())
    }

    /// Returns the CUDA context of this result's device allocations.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.labels.context()
    }

    /// Borrows the completed label image without changing its native layout or values.
    #[must_use]
    pub fn labels(&self) -> &CompressedMarkerLabels {
        &self.labels
    }

    /// Returns the completed contour-pixel image, including its initialized background.
    #[must_use]
    pub fn image(&self) -> &ContiguousImage<u8, C1> {
        &self.pixel_image
    }

    /// Returns the corresponding per-pixel contour-direction image.
    #[must_use]
    pub fn directions(&self) -> &ContiguousImage<ContourPixelDirectionInfo, C1> {
        &self.directions
    }

    /// Returns the logical marker-info prefix ending before `label_count()`.
    ///
    /// # Errors
    ///
    /// Returns an error if the logical label-ID prefix or offset sentinel does not fit within the backing allocation.
    pub fn region_info(&self) -> Result<DeviceView<'_, CompressedMarkerLabelsInfo>> {
        DeviceView::from_memory(&self.info)
            .slice(..self.labels.label_count() as usize)
            .map_err(Into::into)
    }

    /// Initialized native storage, including the size query's reserved working records.
    #[must_use]
    pub fn region_info_storage(&self) -> &DeviceMemory<CompressedMarkerLabelsInfo> {
        &self.info
    }

    /// Returns the completed native totals for this contour result.
    #[must_use]
    pub fn totals(&self) -> &ContourTotalsInfo {
        &self.totals
    }

    /// Returns the host count prefix indexed by this result's label IDs.
    #[must_use]
    pub fn pixel_counts(&self) -> &[u32] {
        &self.counts_host[..self.labels.label_count() as usize]
    }

    /// Returns the host starting-offset prefix, including the final offset sentinel.
    #[must_use]
    pub fn starting_offsets(&self) -> &[u32] {
        &self.offsets_host[..=self.labels.label_count() as usize]
    }

    /// Returns a device view of the logical per-label count prefix.
    ///
    /// # Errors
    ///
    /// Returns an error if the logical label-ID prefix or offset sentinel does not fit within the backing allocation.
    pub fn pixel_counts_device(&self) -> Result<DeviceView<'_, u32>> {
        DeviceView::from_memory(&self.counts)
            .slice(..self.labels.label_count() as usize)
            .map_err(Into::into)
    }

    /// Returns the full count allocation, including native working capacity.
    #[must_use]
    pub fn pixel_counts_storage(&self) -> &DeviceMemory<u32> {
        &self.counts
    }

    /// Returns a device view of the starting-offset prefix and final sentinel.
    ///
    /// # Errors
    ///
    /// Returns an error if the logical label-ID prefix or offset sentinel does not fit within the backing allocation.
    pub fn starting_offsets_device(&self) -> Result<DeviceView<'_, u32>> {
        DeviceView::from_memory(&self.offsets)
            .slice(..=self.labels.label_count() as usize)
            .map_err(Into::into)
    }

    /// Returns the full offset allocation, including native working capacity.
    #[must_use]
    pub fn starting_offsets_storage(&self) -> &DeviceMemory<u32> {
        &self.offsets
    }
}
