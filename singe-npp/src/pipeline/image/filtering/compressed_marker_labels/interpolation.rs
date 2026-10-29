use std::sync::Arc;

use singe_cuda::{
    context::Context as CudaContext,
    memory::{DeviceMemory, MemoryCopyKind, transfer},
    view::{DeviceRepr, DeviceView},
};

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::filtering,
    pipeline::image::filtering::compressed_marker_labels::{
        ContiguousImage,
        geometry::{ContourGeometry, geometry_span},
    },
    types::{Point32f, Point64f, Size},
    utility::checked_len,
};

/// Completed marching-squares coordinates retaining their corresponding contour geometry.
#[derive(Debug)]
pub struct InterpolatedContours<T> {
    geometry: ContourGeometry,
    image: ContiguousImage<T>,
    points: DeviceMemory<T>,
    offsets: Vec<usize>,
}

impl<T: DeviceRepr> InterpolatedContours<T> {
    /// Returns the CUDA context owning the interpolated points and retained geometry.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.points.context()
    }

    /// Returns the interpolated raster dimensions in pixels.
    #[must_use]
    pub fn size(&self) -> Size {
        self.image.size()
    }

    /// Borrows the matching ordered contour geometry retained with these coordinates.
    #[must_use]
    pub fn geometry(&self) -> &ContourGeometry {
        &self.geometry
    }

    /// Interpolated pixels with initialized zero coordinates outside native writes.
    #[must_use]
    pub fn image(&self) -> &ContiguousImage<T> {
        &self.image
    }

    /// Borrows the compact interpolated device point list for one selected label.
    ///
    /// Points follow the completed native contour order. Slots with unset ordered geometry
    /// locations are omitted; a selected label with no retained points yields an empty view.
    /// The view borrows this result and retains its allocation bounds.
    ///
    /// # Errors
    ///
    /// Returns an error if `label` is outside the selected geometry range or constructing
    /// the checked device slice fails.
    pub fn points(&self, label: u32) -> Result<DeviceView<'_, T>> {
        if !self.geometry.labels.contains(&label) {
            return Err(Error::out_of_range("contour label"));
        }
        let index = label as usize;
        let span = self.offsets[index]..self.offsets[index + 1];
        DeviceView::from_memory(&self.points)
            .slice(span)
            .map_err(Into::into)
    }
}

macro_rules! impl_interpolation {
    ($method:ident, $point:ty, $native:path) => {
        impl ContourGeometry {
            /// Completes marching-squares interpolation using this matching native geometry.
            ///
            /// Consumes the retained geometry and allocates a zero-initialized coordinate image
            /// and device point workspace. Uploads ordered host geometry before native interpolation,
            /// then compacts points in contour order, omitting unset ordered locations. The returned
            /// `InterpolatedContours` retains the matching geometry and exposes per-label device
            /// point views. Pixels outside native writes retain zero coordinates.
            /// The supplied stream context must match the geometry context. Rejects capture and
            /// returns after successful native execution and device-copy completion.
            ///
            /// # Errors
            ///
            /// Returns an error if capture checks, context validation, capture-time allocation
            /// retention, geometry upload, checked image or point-count arithmetic, allocation,
            /// image view construction, native interpolation, synchronization, geometry-span
            /// validation, or device-to-device compaction fails.
            pub fn $method(
                mut self,
                stream_context: &StreamContext,
            ) -> Result<InterpolatedContours<$point>> {
                for owner in [
                    self.geometry.allocation(),
                    self.segments.allocation(),
                    self.found.allocation(),
                ] {
                    stream_context
                        .stream()
                        .defer_drop_until_capture_end(Arc::clone(owner))?;
                }
                self.contours.ensure_eager(stream_context)?;
                // NPP orders host geometry after producing its device working records.
                // Marching squares consumes the ordered geometry on the GPU.
                transfer::copy_from_host(&mut self.geometry, &self.geometry_host)?;
                let size = self.size();
                let total = self.contours.totals.total_image_pixel_contour_count;
                let label_count = self.contours.labels.label_count();
                let mut image = ContiguousImage::<$point>::from_buffer(
                    DeviceMemory::zeroed(self.context(), checked_len(size, 1)?)?,
                    size,
                )?;
                let mut points = DeviceMemory::<$point>::zeroed(self.context(), total as usize)?;
                stream_context.sync_scope(|| {
                    // SAFETY: the owner preserves corresponding native working records,
                    // host ordering, counters, offsets, and sparse output backgrounds
                    // through completion, including every error and unwind path.
                    unsafe {
                        $native(
                            stream_context,
                            &self.contours.pixel_image.view()?,
                            &mut image.view_mut()?,
                            &mut self.contours.directions.view_mut()?,
                            &mut self.geometry,
                            &mut self.geometry_host,
                            &mut points,
                            &mut self.found_host,
                            &mut self.contours.offsets,
                            &mut self.contours.offsets_host,
                            total,
                            label_count,
                            self.labels.start,
                            self.labels.end,
                            &mut self.segments,
                            &mut self.segments_host,
                        )
                    }
                })?;
                let (points, offsets) = compact_points(&self, &points, stream_context)?;
                Ok(InterpolatedContours {
                    geometry: self,
                    image,
                    points,
                    offsets,
                })
            }
        }
    };
}

fn compact_points<T: DeviceRepr>(
    geometry: &ContourGeometry,
    source: &DeviceMemory<T>,
    stream_context: &StreamContext,
) -> Result<(DeviceMemory<T>, Vec<usize>)> {
    let count = geometry.contours.labels.label_count() as usize;
    let mut offsets = vec![0; count + 1];
    let mut runs = Vec::new();
    let mut total = 0usize;
    for label in 0..count {
        if geometry
            .labels
            .contains(&(u32::try_from(label).expect("label is bounded by the u32 label count")))
        {
            let span = geometry_span(
                &geometry.contours.offsets_host,
                &geometry.found_host,
                u32::try_from(label).expect("label is bounded by the u32 label count"),
                source.len(),
            )?;
            let mut run = None;
            for index in span.clone() {
                let point = geometry.geometry_host[index].contour_ordered_geometry_location;
                if point.x >= 0 && point.y >= 0 {
                    run.get_or_insert(index);
                    total = total
                        .checked_add(1)
                        .ok_or_else(|| Error::out_of_range("ordered contour points"))?;
                } else if let Some(start) = run.take() {
                    runs.push(start..index);
                }
            }
            if let Some(start) = run {
                runs.push(start..span.end);
            }
        }
        offsets[label + 1] = total;
    }
    let output = DeviceMemory::<T>::create_uninit(geometry.context(), total)?;
    stream_context.sync_scope(|| {
        let mut written = 0;
        for run in &runs {
            // SAFETY: completed native output supplies initialized values for the
            // corresponding ordered host slots. Every disjoint target run remains
            // owned and borrowed through checked completion.
            unsafe {
                DeviceMemory::<T>::copy_async(
                    output.as_mut_ptr().cast::<T>().add(written),
                    source.as_ptr().add(run.start),
                    run.len(),
                    MemoryCopyKind::DeviceToDevice,
                    stream_context.stream(),
                )?;
            }
            written += run.len();
        }
        Ok(())
    })?;
    // SAFETY: the completed disjoint copies initialize exactly the compact allocation.
    Ok((unsafe { output.assume_init() }, offsets))
}

impl_interpolation!(
    interpolate_f32,
    Point32f,
    filtering::contours_image_marching_squares_interpolation_32f_c1
);

impl_interpolation!(
    interpolate_f64,
    Point64f,
    filtering::contours_image_marching_squares_interpolation_64f_c1
);
