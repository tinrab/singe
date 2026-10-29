use std::{mem::size_of, ops::Range, sync::Arc};

use singe_cuda::{context::Context, memory::DeviceMemory, view::DeviceView};

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::filtering,
    pipeline::image::filtering::compressed_marker_labels::Contours,
    types::{
        CompressedMarkerLabelsInfo, ContourBlockSegment, ContourPixelGeometryInfo, Point, Size,
    },
    utility::{checked_len, to_i32},
};

/// Direction of each ordered contour geometry list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContourOrder {
    /// Orders each selected contour clockwise.
    Clockwise,
    /// Orders each selected contour counterclockwise.
    Counterclockwise,
}

impl Contours {
    /// Produces ordered geometry for a half-open range of native label IDs.
    /// The range must be nonempty and end at or before `labels().label_count()`.
    /// This consumes the corresponding native metadata, which geometry generation updates in place.
    ///
    /// # Errors
    ///
    /// Returns an error during capture, if contexts differ, or if `labels` is empty, reversed,
    /// or ends beyond the native label count. Scratch-size queries, byte-count conversions,
    /// image-size arithmetic, allocation or readback, native geometry generation, stream
    /// synchronization, and validating generated geometry spans can also fail.
    pub fn geometry(
        mut self,
        stream_context: &StreamContext,
        labels: Range<u32>,
        order: ContourOrder,
    ) -> Result<ContourGeometry> {
        self.ensure_eager(stream_context)?;
        if labels.start >= labels.end || labels.end > self.labels.label_count() {
            return Err(Error::out_of_range("contour label range"));
        }
        let total = self.totals.total_image_pixel_contour_count;
        let count = self.labels.label_count();
        let geometry_count = self.offsets_host[count as usize];
        let geometry_len =
            filtering::compressed_marker_labels_uf_geometry_lists_size_c1(geometry_count)?
                .div_ceil(size_of::<ContourPixelGeometryInfo>());
        let segment_len =
            filtering::compressed_marker_labels_uf_contours_block_segment_list_size_c1(
                &mut self.counts_host,
                total,
                count,
                labels.start,
                labels.end,
            )?
            .div_ceil(size_of::<ContourBlockSegment>());
        let context = self.context();
        let mut info_host = self.info.copy_to_host_vec()?;
        let mut geometry = DeviceMemory::<ContourPixelGeometryInfo>::zeroed(context, geometry_len)?;
        let mut geometry_host = vec![ContourPixelGeometryInfo::default(); geometry_len];
        let mut segments = DeviceMemory::<ContourBlockSegment>::zeroed(context, segment_len)?;
        let mut segments_host = vec![ContourBlockSegment::default(); segment_len];
        let mut found = DeviceMemory::<u32>::zeroed(context, self.counts_host.len())?;
        let mut found_host = vec![0; self.counts_host.len()];
        let size = self.labels.size();
        let mut image = vec![0; checked_len(size, 1)?];
        let step = to_i32(size.width as usize, "contour image step")?;
        stream_context.sync_scope(|| {
            // SAFETY: all inputs belong to this native contour-information call,
            // query-sized initialized storage covers sparse updates, and every owner
            // and host borrow survives native completion.
            unsafe {
                filtering::compressed_marker_labels_uf_contours_generate_geometry_lists_c1(
                    stream_context,
                    &mut self.info,
                    &mut info_host,
                    &mut self.directions.view_mut()?,
                    &mut geometry,
                    &mut geometry_host,
                    &mut image,
                    step,
                    &mut self.counts,
                    &mut found,
                    &mut found_host,
                    &mut self.offsets,
                    &mut self.offsets_host,
                    total,
                    count,
                    labels.start,
                    labels.end,
                    &mut segments,
                    &mut segments_host,
                    order == ContourOrder::Counterclockwise,
                )
            }
        })?;
        for label in labels.clone() {
            geometry_span(&self.offsets_host, &found_host, label, geometry_len)?;
        }
        Ok(ContourGeometry {
            contours: self,
            labels,
            info_host,
            geometry,
            geometry_host,
            segments,
            segments_host,
            found,
            found_host,
            image,
        })
    }
}

/// Completed ordered contours retaining their corresponding native metadata and allocation owners.
#[derive(Debug)]
pub struct ContourGeometry {
    pub(crate) contours: Contours,
    pub(crate) labels: Range<u32>,
    info_host: Vec<CompressedMarkerLabelsInfo>,
    pub(crate) geometry: DeviceMemory<ContourPixelGeometryInfo>,
    pub(crate) geometry_host: Vec<ContourPixelGeometryInfo>,
    pub(crate) segments: DeviceMemory<ContourBlockSegment>,
    pub(crate) segments_host: Vec<ContourBlockSegment>,
    pub(crate) found: DeviceMemory<u32>,
    pub(crate) found_host: Vec<u32>,
    image: Vec<u8>,
}

impl ContourGeometry {
    /// Returns the CUDA context shared by the retained contour allocations.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.contours.context()
    }

    /// Returns the source label-image dimensions in pixels.
    #[must_use]
    pub fn size(&self) -> Size {
        self.contours.labels.size()
    }

    /// Returns the selected half-open range of native label IDs.
    #[must_use]
    pub fn label_range(&self) -> Range<u32> {
        self.labels.clone()
    }

    /// Borrows the retained contour metadata after ordered geometry generation.
    #[must_use]
    pub fn contours(&self) -> &Contours {
        &self.contours
    }

    /// Returns the compact row-major host contour raster for the selected label range.
    #[must_use]
    pub fn image(&self) -> &[u8] {
        &self.image
    }

    /// Returns the host native region-information array indexed by label ID.
    ///
    /// The selected entries include updates made during ordered geometry generation.
    #[must_use]
    pub fn region_info(&self) -> &[CompressedMarkerLabelsInfo] {
        &self.info_host[..self.contours.labels.label_count() as usize]
    }

    /// Returns the completed host records for this selected contour.
    /// Only `contour_ordered_geometry_location` describes each ordered point;
    /// the other fields retain NPP's working metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if `label` lies outside the selected range, offset or count tables have
    /// no entry for it, or the resulting span overflows or exceeds the host geometry capacity.
    pub fn records(&self, label: u32) -> Result<&[ContourPixelGeometryInfo]> {
        if !self.labels.contains(&label) {
            return Err(Error::out_of_range("contour label"));
        }
        let span = geometry_span(
            &self.contours.offsets_host,
            &self.found_host,
            label,
            self.geometry_host.len(),
        )?;
        Ok(&self.geometry_host[span])
    }

    /// Iterates the ordered host point list produced by NPP.
    /// Native unset locations remain visible through `records` and are omitted here.
    ///
    /// # Errors
    ///
    /// Returns an error if selecting the host geometry records for `label` fails.
    pub fn points(&self, label: u32) -> Result<impl Iterator<Item = Point> + '_> {
        Ok(self
            .records(label)?
            .iter()
            .map(|record| record.contour_ordered_geometry_location)
            .filter(|point| point.x >= 0 && point.y >= 0))
    }

    /// Completes an upload of the ordered point list into an independent allocation.
    /// NPP orders the host records while retaining different device working records.
    ///
    /// # Errors
    ///
    /// Returns an error if selecting points for `label`, allocating independent device storage,
    /// or uploading the points fails.
    pub fn copy_points_to_device(&self, label: u32) -> Result<DeviceMemory<Point>> {
        DeviceMemory::from_slice(self.context(), &self.points(label)?.collect::<Vec<_>>())
            .map_err(Into::into)
    }

    /// Borrows native device working records, which are not the ordered host output.
    ///
    /// # Errors
    ///
    /// Returns an error if `label` is outside the selected range, its offset or count is missing,
    /// or its span overflows or exceeds the device geometry capacity. Constructing or slicing
    /// the device view can also fail.
    pub fn working_records(&self, label: u32) -> Result<DeviceView<'_, ContourPixelGeometryInfo>> {
        if !self.labels.contains(&label) {
            return Err(Error::out_of_range("contour label"));
        }
        let span = geometry_span(
            &self.contours.offsets_host,
            &self.found_host,
            label,
            self.geometry.len(),
        )?;
        DeviceView::from_memory(&self.geometry)
            .slice(span)
            .map_err(Into::into)
    }

    /// Returns host geometry counts indexed by native label ID.
    #[must_use]
    pub fn pixels_found(&self) -> &[u32] {
        &self.found_host[..self.contours.labels.label_count() as usize]
    }

    /// Borrows the device geometry counts up to the native label count.
    ///
    /// # Errors
    ///
    /// Returns an error if the retained device count allocation cannot cover the native
    /// label-count range.
    pub fn pixels_found_device(&self) -> Result<DeviceView<'_, u32>> {
        DeviceView::from_memory(&self.found)
            .slice(..self.contours.labels.label_count() as usize)
            .map_err(Into::into)
    }

    /// Initialized native segment storage, including reserved working slots.
    #[must_use]
    pub fn segment_storage(&self) -> &DeviceMemory<ContourBlockSegment> {
        &self.segments
    }

    /// Returns the host native segment storage, including reserved working slots.
    #[must_use]
    pub fn segment_storage_host(&self) -> &[ContourBlockSegment] {
        &self.segments_host
    }
}

/// Returns the half-open geometry span for `label` using its offset and count.
///
/// # Errors
///
/// Returns an error if either table has no entry for `label`, offset plus count overflows,
/// or the resulting end exceeds `capacity`.
pub fn geometry_span(
    offsets: &[u32],
    counts: &[u32],
    label: u32,
    capacity: usize,
) -> Result<Range<usize>> {
    let index = label as usize;
    let start = offsets
        .get(index)
        .copied()
        .ok_or_else(|| Error::out_of_range("contour label"))? as usize;
    let count = counts
        .get(index)
        .copied()
        .ok_or_else(|| Error::out_of_range("contour label"))? as usize;
    let end = start
        .checked_add(count)
        .filter(|&end| end <= capacity)
        .ok_or_else(|| Error::out_of_range("produced contour geometry"))?;
    Ok(start..end)
}
