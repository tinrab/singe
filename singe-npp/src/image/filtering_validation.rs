use std::{cmp::Ordering, mem::size_of};

use singe_cuda::memory::DeviceMemory;
use singe_npp_sys as sys;

use crate::{
    error::{Error, Result},
    image::{
        filtering::{
            compress_marker_labels_uf_buffer_size_u32_c1,
            compressed_marker_labels_uf_contours_block_segment_list_size_c1,
        },
        view::{
            C1, ChannelLayout, ImageView, ImageViewMut, image_descriptors, image_descriptors_mut,
        },
    },
    types::{
        BorderType, CompressedMarkerLabelsInfo, ContourBlockSegment, ContourPixelGeometryInfo,
        MaskSize, Point, PointPolar, Size,
    },
    utility::batch_size_i32,
    workspace::{BufferDescriptors, create_buffer_descriptors, create_repeated_buffer_descriptors},
};

pub(super) fn validate_same_size(source: Size, destination: Size) -> Result<()> {
    if source == destination {
        return Ok(());
    }
    Err(Error::SizeMismatch {
        name: "image size".into(),
        expected: source,
        actual: destination,
    })
}

pub(super) fn validate_filter_kernel_any<T>(kernel: &[T], anchor: i32) -> Result<()> {
    let mask_size = i32::try_from(kernel.len()).map_err(|_| Error::OutOfRange {
        name: "kernel length".into(),
    })?;
    if mask_size <= 0 {
        return Err(Error::OutOfRange {
            name: "kernel length".into(),
        });
    }
    if (0..mask_size).contains(&anchor) {
        return Ok(());
    }

    Err(Error::OutOfRange {
        name: "anchor".into(),
    })
}

pub(super) fn validate_filter_kernel(kernel: &[i32], anchor: i32) -> Result<()> {
    validate_filter_kernel_any(kernel, anchor)
}

pub(super) fn validate_filter_kernel_f32(kernel: &[f32], anchor: i32) -> Result<()> {
    validate_filter_kernel_any(kernel, anchor)
}

pub(super) fn validate_filter_axis_mask(mask_size: i32, anchor: i32) -> Result<()> {
    if mask_size <= 0 {
        return Err(Error::OutOfRange {
            name: "mask size".into(),
        });
    }
    if (0..mask_size).contains(&anchor) {
        return Ok(());
    }

    Err(Error::OutOfRange {
        name: "anchor".into(),
    })
}

pub(super) fn validate_advanced_filter_kernel(kernel: &[f32]) -> Result<()> {
    if kernel.is_empty() {
        return Err(Error::OutOfRange {
            name: "kernel length".into(),
        });
    }

    Ok(())
}

pub(super) fn validate_gauss_pyramid_rate(rate: f32) -> Result<()> {
    if rate > 1.0 && rate <= 10.0 {
        return Ok(());
    }

    Err(Error::OutOfRange {
        name: "rate".into(),
    })
}

pub(super) fn validate_bilateral_gauss_parameters(
    radius: i32,
    step_between_source_pixels: i32,
    value_square_sigma: f32,
    position_square_sigma: f32,
    border_type: BorderType,
) -> Result<()> {
    if !(1..=32).contains(&radius) {
        return Err(Error::OutOfRange {
            name: "radius".into(),
        });
    }
    if step_between_source_pixels <= 0 {
        return Err(Error::OutOfRange {
            name: "step between source pixels".into(),
        });
    }
    if value_square_sigma.partial_cmp(&0.0) != Some(Ordering::Greater) {
        return Err(Error::OutOfRange {
            name: "value square sigma".into(),
        });
    }
    if position_square_sigma.partial_cmp(&0.0) != Some(Ordering::Greater) {
        return Err(Error::OutOfRange {
            name: "position square sigma".into(),
        });
    }
    if border_type != BorderType::Replicate {
        return Err(Error::OutOfRange {
            name: "border type".into(),
        });
    }

    Ok(())
}

pub(super) fn validate_unsharp_parameters(radius: f32, sigma: f32) -> Result<()> {
    if radius.partial_cmp(&0.0) != Some(Ordering::Greater) {
        return Err(Error::OutOfRange {
            name: "radius".into(),
        });
    }
    if sigma.partial_cmp(&0.0) != Some(Ordering::Greater) {
        return Err(Error::OutOfRange {
            name: "sigma".into(),
        });
    }

    Ok(())
}

pub(super) fn validate_positive_size(size: Size) -> Result<()> {
    if size.width > 0 && size.height > 0 {
        return Ok(());
    }

    Err(Error::OutOfRange {
        name: "size".into(),
    })
}

pub(super) fn validate_source_offset(source: Size, source_offset: Point) -> Result<()> {
    if source_offset.x < 0 || source_offset.y < 0 {
        return Err(Error::OutOfRange {
            name: "source offset".into(),
        });
    }
    if source_offset.x >= source.width || source_offset.y >= source.height {
        return Err(Error::OutOfRange {
            name: "source offset".into(),
        });
    }

    Ok(())
}

pub(super) fn validate_seed(size: Size, seed: Point) -> Result<()> {
    validate_positive_size(size)?;
    if seed.x >= 0 && seed.y >= 0 && seed.x < size.width && seed.y < size.height {
        return Ok(());
    }

    Err(Error::OutOfRange {
        name: "seed".into(),
    })
}

pub(super) fn validate_filter_mask(mask_size: Size, anchor: Point) -> Result<()> {
    if mask_size.width <= 0 || mask_size.height <= 0 {
        return Err(Error::OutOfRange {
            name: "mask size".into(),
        });
    }
    if anchor.x < 0 || anchor.y < 0 || anchor.x >= mask_size.width || anchor.y >= mask_size.height {
        return Err(Error::OutOfRange {
            name: "anchor".into(),
        });
    }

    Ok(())
}

pub(super) fn validate_filter_kernel_2d<T>(
    kernel: &[T],
    kernel_size: Size,
    anchor: Point,
) -> Result<()> {
    validate_filter_mask(kernel_size, anchor)?;
    let width = usize::try_from(kernel_size.width).map_err(|_| Error::OutOfRange {
        name: "kernel width".into(),
    })?;
    let height = usize::try_from(kernel_size.height).map_err(|_| Error::OutOfRange {
        name: "kernel height".into(),
    })?;
    let expected = width.checked_mul(height).ok_or_else(|| Error::OutOfRange {
        name: "kernel length".into(),
    })?;
    if kernel.len() == expected {
        return Ok(());
    }

    Err(Error::OutOfRange {
        name: "kernel length".into(),
    })
}

pub(super) fn validate_border_roi(source: Size, source_offset: Point, roi: Size) -> Result<()> {
    if source_offset.x < 0 || source_offset.y < 0 {
        return Err(Error::OutOfRange {
            name: "source offset".into(),
        });
    }

    let right = source_offset
        .x
        .checked_add(roi.width)
        .ok_or_else(|| Error::OutOfRange {
            name: "source offset".into(),
        })?;
    let bottom = source_offset
        .y
        .checked_add(roi.height)
        .ok_or_else(|| Error::OutOfRange {
            name: "source offset".into(),
        })?;

    if right <= source.width && bottom <= source.height {
        return Ok(());
    }

    Err(Error::SizeMismatch {
        name: "border roi".into(),
        expected: source,
        actual: roi,
    })
}

pub(super) fn validate_canny_thresholds(low_threshold: i16, high_threshold: i16) -> Result<()> {
    if low_threshold < 0 || high_threshold < 0 {
        return Err(Error::OutOfRange {
            name: "threshold".into(),
        });
    }
    if low_threshold <= high_threshold {
        return Ok(());
    }
    Err(Error::LengthMismatch {
        name: "threshold range".into(),
        expected: low_threshold as usize,
        actual: high_threshold as usize,
    })
}

pub(super) fn validate_harris_corners_border_parameters(
    mask_size: MaskSize,
    average_window_size: MaskSize,
    border_type: BorderType,
) -> Result<()> {
    match mask_size {
        MaskSize::Mask3x3 | MaskSize::Mask5x5 => {}
        _ => {
            return Err(Error::OutOfRange {
                name: "mask size".into(),
            });
        }
    }
    match average_window_size {
        MaskSize::Mask3x3 | MaskSize::Mask5x5 => {}
        _ => {
            return Err(Error::OutOfRange {
                name: "average window size".into(),
            });
        }
    }
    if border_type == BorderType::Replicate {
        return Ok(());
    }
    Err(Error::OutOfRange {
        name: "border type".into(),
    })
}

pub(super) fn validate_hough_line_parameters(delta: PointPolar, max_line_count: i32) -> Result<()> {
    if !(delta.rho > 0.0 && delta.rho < 3.0) {
        return Err(Error::OutOfRange {
            name: "delta rho".into(),
        });
    }
    if !(delta.theta > 0.25 && delta.theta < 3.0) {
        return Err(Error::OutOfRange {
            name: "delta theta".into(),
        });
    }
    if max_line_count > 0 {
        return Ok(());
    }
    Err(Error::OutOfRange {
        name: "max line count".into(),
    })
}

pub(super) fn validate_hough_line_io(
    roi: Size,
    delta: PointPolar,
    threshold: i32,
    lines: &DeviceMemory<PointPolar>,
    line_count: &DeviceMemory<i32>,
) -> Result<()> {
    let max_line_count = i32::try_from(lines.len()).map_err(|_| Error::OutOfRange {
        name: "max line count".into(),
    })?;
    validate_positive_size(roi)?;
    validate_hough_line_parameters(delta, max_line_count)?;
    if threshold <= 0 {
        return Err(Error::OutOfRange {
            name: "threshold".into(),
        });
    }
    if line_count.len() == 1 {
        return Ok(());
    }
    Err(Error::LengthMismatch {
        name: "line count".into(),
        expected: 1,
        actual: line_count.len(),
    })
}

pub(super) fn validate_hough_line_destination_roi(destination_roi: [PointPolar; 2]) -> Result<()> {
    if destination_roi[0].rho <= destination_roi[1].rho
        && destination_roi[0].theta <= destination_roi[1].theta
    {
        return Ok(());
    }
    Err(Error::OutOfRange {
        name: "destination roi".into(),
    })
}

pub(super) fn optional_image_output<T, L>(
    output: Option<&mut ImageViewMut<'_, T, L>>,
) -> (*mut T, i32, Option<Size>)
where
    L: ChannelLayout,
{
    match output {
        Some(image) => (image.as_mut_ptr().cast(), image.step(), Some(image.size())),
        None => (std::ptr::null_mut(), 0, None),
    }
}

pub(super) fn validate_gradient_vector_outputs(
    destination_x: Option<Size>,
    destination_y: Option<Size>,
    destination_magnitude: Option<Size>,
    destination_angle: Option<Size>,
) -> Result<Size> {
    let roi = destination_x
        .or(destination_y)
        .or(destination_magnitude)
        .or(destination_angle)
        .ok_or_else(|| Error::OutOfRange {
            name: "gradient destination".into(),
        })?;
    validate_positive_size(roi)?;

    for destination in [
        destination_x,
        destination_y,
        destination_magnitude,
        destination_angle,
    ]
    .into_iter()
    .flatten()
    {
        validate_same_size(roi, destination)?;
    }

    Ok(roi)
}

pub(super) fn validate_distance_transform_outputs(
    destination_voronoi: Option<Size>,
    destination_voronoi_indices: Option<Size>,
    destination_voronoi_manhattan: Option<Size>,
    destination_transform: Option<Size>,
) -> Result<Size> {
    let roi = destination_voronoi
        .or(destination_voronoi_indices)
        .or(destination_voronoi_manhattan)
        .or(destination_transform)
        .ok_or_else(|| Error::OutOfRange {
            name: "distance transform destination".into(),
        })?;
    validate_positive_size(roi)?;

    for destination in [
        destination_voronoi,
        destination_voronoi_indices,
        destination_voronoi_manhattan,
        destination_transform,
    ]
    .into_iter()
    .flatten()
    {
        validate_same_size(roi, destination)?;
    }

    Ok(roi)
}

pub(super) fn validate_label_marker_step(
    source_destination: &ImageViewMut<'_, u32, C1>,
) -> Result<()> {
    validate_u32_label_marker_step(source_destination.size(), source_destination.step())
}

pub(super) fn validate_u32_label_marker_step(size: Size, step: i32) -> Result<()> {
    let expected = usize::try_from(size.width)
        .ok()
        .and_then(|width| width.checked_mul(size_of::<u32>()))
        .ok_or_else(|| Error::OutOfRange {
            name: "source destination step".into(),
        })?;
    let actual = usize::try_from(step).map_err(|_| Error::OutOfRange {
        name: "source destination step".into(),
    })?;

    if actual == expected {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: "source destination step".into(),
        expected,
        actual,
    })
}

pub(super) fn validate_marker_label_info_output(
    output: &DeviceMemory<CompressedMarkerLabelsInfo>,
    max_marker_label_id: u32,
) -> Result<()> {
    let expected = usize::try_from(max_marker_label_id).map_err(|_| Error::OutOfRange {
        name: "max marker label id".into(),
    })?;
    if output.len() >= expected {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: "marker labels info list".into(),
        expected,
        actual: output.len(),
    })
}

pub(super) fn validate_marker_label_info_host_output(
    output: &[CompressedMarkerLabelsInfo],
    max_marker_label_id: u32,
) -> Result<()> {
    let expected = usize::try_from(max_marker_label_id).map_err(|_| Error::OutOfRange {
        name: "max marker label id".into(),
    })?;
    if output.len() >= expected {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: "marker labels info list".into(),
        expected,
        actual: output.len(),
    })
}

pub(super) fn validate_u32_list_dev_output(
    output_dev: &DeviceMemory<u32>,
    max_marker_label_id: u32,
    name: &str,
) -> Result<()> {
    let expected = usize::try_from(max_marker_label_id).map_err(|_| Error::OutOfRange {
        name: "max marker label id".into(),
    })?;
    if output_dev.len() >= expected {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: name.into(),
        expected,
        actual: output_dev.len(),
    })
}

pub(super) fn validate_u32_list_host_output(
    output_host: &[u32],
    count: u32,
    name: &str,
) -> Result<()> {
    let expected = usize::try_from(count).map_err(|_| Error::OutOfRange { name: name.into() })?;
    if output_host.len() >= expected {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: name.into(),
        expected,
        actual: output_host.len(),
    })
}

pub(super) fn validate_u32_list_device_and_host_output(
    output_dev: &DeviceMemory<u32>,
    output_host: &[u32],
    count: u32,
    name: &str,
) -> Result<()> {
    validate_u32_list_dev_output(output_dev, count, name)?;
    validate_u32_list_host_output(output_host, count, name)
}

pub(super) fn validate_u32_list_output(
    output_dev: &DeviceMemory<u32>,
    output_host: &[u32],
    max_marker_label_id: u32,
    name: &str,
) -> Result<()> {
    let expected = usize::try_from(max_marker_label_id).map_err(|_| Error::OutOfRange {
        name: "max marker label id".into(),
    })?;
    if output_dev.len() < expected {
        return Err(Error::LengthMismatch {
            name: name.into(),
            expected,
            actual: output_dev.len(),
        });
    }
    if output_host.len() < expected {
        return Err(Error::LengthMismatch {
            name: name.into(),
            expected,
            actual: output_host.len(),
        });
    }
    Ok(())
}

pub(super) fn validate_typed_output_len(
    actual: usize,
    expected_count: u32,
    name: &str,
) -> Result<()> {
    let expected =
        usize::try_from(expected_count).map_err(|_| Error::OutOfRange { name: name.into() })?;
    if actual == expected {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: name.into(),
        expected,
        actual,
    })
}

pub(super) fn validate_geometry_list_id_range(
    first: u32,
    last: u32,
    max_marker_label_id: u32,
) -> Result<()> {
    if first >= last {
        return Err(Error::LengthMismatch {
            name: "contour geometry list range".into(),
            expected: usize::try_from(first)
                .unwrap_or(usize::MAX)
                .saturating_add(1),
            actual: usize::try_from(last).unwrap_or(usize::MAX),
        });
    }

    if last <= max_marker_label_id {
        return Ok(());
    }

    Err(Error::OutOfRange {
        name: "last contour geometry list id".into(),
    })
}

pub(super) fn validate_geometry_list_buffers(
    output_dev: &DeviceMemory<ContourPixelGeometryInfo>,
    output_host: &[ContourPixelGeometryInfo],
) -> Result<()> {
    if output_dev.len() == output_host.len() {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: "contours pixel geometry lists".into(),
        expected: output_host.len(),
        actual: output_dev.len(),
    })
}

pub(super) fn validate_contours_geometry_image_host(
    output: &[u8],
    size: Size,
    step: i32,
) -> Result<()> {
    let step = usize::try_from(step).map_err(|_| Error::OutOfRange {
        name: "contours geometry image step".into(),
    })?;
    let width = usize::try_from(size.width).map_err(|_| Error::OutOfRange {
        name: "contours geometry image width".into(),
    })?;
    let height = usize::try_from(size.height).map_err(|_| Error::OutOfRange {
        name: "contours geometry image height".into(),
    })?;
    let required = if height == 0 {
        0
    } else {
        step.checked_mul(height - 1)
            .and_then(|bytes| bytes.checked_add(width))
            .ok_or_else(|| Error::OutOfRange {
                name: "contours geometry image".into(),
            })?
    };

    if output.len() >= required {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: "contours geometry image".into(),
        expected: required,
        actual: output.len(),
    })
}

pub(super) fn validate_contours_block_segment_output(
    output_dev: &DeviceMemory<ContourBlockSegment>,
    output_host: &[ContourBlockSegment],
    contours_pixel_counts_list_host: &mut [u32],
    total_image_pixel_contour_count: u32,
    compressed_label_count: u32,
    first_contour_geometry_list_id: u32,
    last_contour_geometry_list_id: u32,
) -> Result<()> {
    let required_bytes = compressed_marker_labels_uf_contours_block_segment_list_size_c1(
        contours_pixel_counts_list_host,
        total_image_pixel_contour_count,
        compressed_label_count,
        first_contour_geometry_list_id,
        last_contour_geometry_list_id,
    )?;
    let required = required_bytes
        .checked_div(size_of::<ContourBlockSegment>())
        .ok_or_else(|| Error::OutOfRange {
            name: "contours block segment list".into(),
        })?;

    if output_dev.len() != required {
        return Err(Error::LengthMismatch {
            name: "contours block segment list".into(),
            expected: required,
            actual: output_dev.len(),
        });
    }
    if output_host.len() != required {
        return Err(Error::LengthMismatch {
            name: "contours block segment list".into(),
            expected: required,
            actual: output_host.len(),
        });
    }

    Ok(())
}

pub(super) fn validate_block_segment_inputs(
    output_dev: &DeviceMemory<ContourBlockSegment>,
    output_host: &[ContourBlockSegment],
) -> Result<()> {
    if output_dev.len() == output_host.len() {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: "contours block segment list".into(),
        expected: output_host.len(),
        actual: output_dev.len(),
    })
}

pub(super) fn validate_equal_size(images: &[ImageViewMut<'_, u32, C1>]) -> Result<Size> {
    let Some(first) = images.first() else {
        return Err(Error::OutOfRange {
            name: "batch size".into(),
        });
    };
    let size = first.size();
    for image in &images[1..] {
        validate_same_size(size, image.size())?;
    }
    Ok(size)
}

pub(super) fn label_markers_uf_batch_descriptors<T>(
    sources: &[ImageView<'_, T, C1>],
    destinations: &mut [ImageViewMut<'_, u32, C1>],
) -> Result<(
    i32,
    Size,
    DeviceMemory<sys::NppiImageDescriptor>,
    DeviceMemory<sys::NppiImageDescriptor>,
)> {
    let batch_size = label_markers_uf_batch_size(sources.len(), destinations.len())?;

    for (source, destination) in sources.iter().zip(destinations.iter_mut()) {
        validate_same_size(source.size(), destination.size())?;
    }

    let roi = smallest_size(
        sources
            .iter()
            .map(ImageView::size)
            .chain(destinations.iter().map(ImageViewMut::size)),
    )?;
    let source_descriptors = image_descriptors(sources);
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        roi,
        DeviceMemory::from_slice(&source_descriptors)?,
        DeviceMemory::from_slice(&destination_descriptors)?,
    ))
}

pub(super) fn label_markers_uf_batch_advanced_descriptors<T>(
    sources: &[ImageView<'_, T, C1>],
    destinations: &mut [ImageViewMut<'_, u32, C1>],
) -> Result<(
    i32,
    Size,
    DeviceMemory<sys::NppiImageDescriptor>,
    DeviceMemory<sys::NppiImageDescriptor>,
)> {
    let batch_size = label_markers_uf_batch_size(sources.len(), destinations.len())?;

    for (source, destination) in sources.iter().zip(destinations.iter_mut()) {
        validate_same_size(source.size(), destination.size())?;
    }

    let max_roi = largest_size(destinations.iter().map(ImageViewMut::size))?;
    let source_descriptors = image_descriptors(sources);
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        max_roi,
        DeviceMemory::from_slice(&source_descriptors)?,
        DeviceMemory::from_slice(&destination_descriptors)?,
    ))
}

pub(super) fn compress_marker_labels_uf_batch_descriptors(
    source_destinations: &mut [ImageViewMut<'_, u32, C1>],
) -> Result<(i32, Size, DeviceMemory<sys::NppiImageDescriptor>)> {
    let batch_size =
        label_markers_uf_batch_size(source_destinations.len(), source_destinations.len())?;
    let roi = validate_equal_size(source_destinations)?;
    for image in source_destinations.iter() {
        validate_label_marker_step(image)?;
    }

    let descriptors = image_descriptors_mut(source_destinations);

    Ok((batch_size, roi, DeviceMemory::from_slice(&descriptors)?))
}

pub(super) fn compress_marker_labels_uf_batch_advanced_descriptors(
    source_destinations: &mut [ImageViewMut<'_, u32, C1>],
) -> Result<(i32, Size, DeviceMemory<sys::NppiImageDescriptor>)> {
    let batch_size =
        label_markers_uf_batch_size(source_destinations.len(), source_destinations.len())?;
    for image in source_destinations.iter() {
        validate_label_marker_step(image)?;
    }

    let max_roi = largest_size(source_destinations.iter().map(ImageViewMut::size))?;
    let descriptors = image_descriptors_mut(source_destinations);

    Ok((batch_size, max_roi, DeviceMemory::from_slice(&descriptors)?))
}

pub(super) fn compress_marker_labels_uf_batch_buffers(
    batch_size: usize,
    bytes_per_image: usize,
) -> Result<(Vec<DeviceMemory<u8>>, BufferDescriptors)> {
    create_repeated_buffer_descriptors(batch_size, bytes_per_image)
}

pub(super) fn compress_marker_labels_uf_batch_advanced_buffers(
    source_destinations: &[ImageViewMut<'_, u32, C1>],
) -> Result<(usize, Vec<DeviceMemory<u8>>, BufferDescriptors)> {
    let mut largest_per_image_bytes = 0_usize;
    let mut buffer_sizes = Vec::with_capacity(source_destinations.len());

    for image in source_destinations {
        let per_image_bytes =
            compress_marker_labels_uf_buffer_size_u32_c1(roi_area(image.size())?)?;
        largest_per_image_bytes = largest_per_image_bytes.max(per_image_bytes);
        buffer_sizes.push(per_image_bytes);
    }
    let (buffers, descriptors) = create_buffer_descriptors(buffer_sizes)?;

    Ok((largest_per_image_bytes, buffers, descriptors))
}

pub(super) fn validate_batch_label_output(
    output: &DeviceMemory<u32>,
    batch_size: usize,
) -> Result<()> {
    if output.len() == batch_size {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: "batch label output".into(),
        expected: batch_size,
        actual: output.len(),
    })
}

pub(super) fn roi_area(roi: Size) -> Result<i32> {
    roi.width
        .checked_mul(roi.height)
        .ok_or_else(|| Error::OutOfRange {
            name: "starting number".into(),
        })
}

pub(super) fn label_markers_uf_batch_size(
    source_len: usize,
    destination_len: usize,
) -> Result<i32> {
    batch_size_i32(source_len, destination_len)
}

pub(super) fn smallest_size(sizes: impl IntoIterator<Item = Size>) -> Result<Size> {
    sizes
        .into_iter()
        .reduce(|left, right| Size {
            width: left.width.min(right.width),
            height: left.height.min(right.height),
        })
        .ok_or_else(|| Error::OutOfRange {
            name: "batch size".into(),
        })
}

pub(super) fn largest_size(sizes: impl IntoIterator<Item = Size>) -> Result<Size> {
    sizes
        .into_iter()
        .reduce(|left, right| Size {
            width: left.width.max(right.width),
            height: left.height.max(right.height),
        })
        .ok_or_else(|| Error::OutOfRange {
            name: "batch size".into(),
        })
}
