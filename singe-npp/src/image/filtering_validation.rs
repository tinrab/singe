use std::{
    cmp::Ordering,
    mem::{MaybeUninit, size_of},
    ptr,
    sync::Arc,
};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, view::DeviceOutput};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::{
        filtering::{
            compress_marker_labels_uf_buffer_size_u32_c1,
            compressed_marker_labels_uf_geometry_lists_size_c1,
            compressed_marker_labels_uf_info_list_size_u32_c1,
        },
        view::{
            C1, ChannelLayout, ImageOutputView, ImageView, ImageViewMut, image_descriptors,
            image_descriptors_mut,
        },
    },
    types::{
        BorderType, CompressedMarkerLabelsInfo, ContourBlockSegment, ContourPixelGeometryInfo,
        MaskSize, Point, PointPolar, Size,
    },
    utility::batch_size_i32,
    workspace::{BufferDescriptors, create_buffer_descriptors, create_repeated_buffer_descriptors},
};

pub fn validate_same_size(source: Size, destination: Size) -> Result<()> {
    if source == destination {
        return Ok(());
    }
    Err(Error::SizeMismatch {
        name: "image size".into(),
        expected: source,
        actual: destination,
    })
}

pub fn validate_filter_kernel_any<T>(kernel: &[T], anchor: i32) -> Result<()> {
    let mask_size =
        i32::try_from(kernel.len()).map_err(|_| Error::out_of_range("kernel length"))?;
    if mask_size <= 0 {
        return Err(Error::out_of_range("kernel length"));
    }
    if (0..mask_size).contains(&anchor) {
        return Ok(());
    }

    Err(Error::out_of_range("anchor"))
}

pub fn validate_filter_kernel(kernel: &[i32], anchor: i32) -> Result<()> {
    validate_filter_kernel_any(kernel, anchor)
}

pub fn validate_filter_kernel_f32(kernel: &[f32], anchor: i32) -> Result<()> {
    validate_filter_kernel_any(kernel, anchor)
}

pub fn validate_filter_axis_mask(mask_size: i32, anchor: i32) -> Result<()> {
    if mask_size <= 0 {
        return Err(Error::out_of_range("mask size"));
    }
    if (0..mask_size).contains(&anchor) {
        return Ok(());
    }

    Err(Error::out_of_range("anchor"))
}

pub fn validate_advanced_filter_kernel(kernel: &[f32]) -> Result<()> {
    if kernel.is_empty() {
        return Err(Error::out_of_range("kernel length"));
    }

    Ok(())
}

pub fn validate_gauss_pyramid_rate(rate: f32) -> Result<()> {
    if rate > 1.0 && rate <= 10.0 {
        return Ok(());
    }

    Err(Error::out_of_range("rate"))
}

pub fn validate_bilateral_gauss_parameters(
    radius: i32,
    step_between_source_pixels: i32,
    value_square_sigma: f32,
    position_square_sigma: f32,
    border_type: BorderType,
) -> Result<()> {
    if !(1..=32).contains(&radius) {
        return Err(Error::out_of_range("radius"));
    }
    if step_between_source_pixels <= 0 {
        return Err(Error::out_of_range("step between source pixels"));
    }
    if value_square_sigma.partial_cmp(&0.0) != Some(Ordering::Greater) {
        return Err(Error::out_of_range("value square sigma"));
    }
    if position_square_sigma.partial_cmp(&0.0) != Some(Ordering::Greater) {
        return Err(Error::out_of_range("position square sigma"));
    }
    if border_type != BorderType::Replicate {
        return Err(Error::out_of_range("border type"));
    }

    Ok(())
}

pub fn validate_unsharp_parameters(radius: f32, sigma: f32) -> Result<()> {
    if radius.partial_cmp(&0.0) != Some(Ordering::Greater) {
        return Err(Error::out_of_range("radius"));
    }
    if sigma.partial_cmp(&0.0) != Some(Ordering::Greater) {
        return Err(Error::out_of_range("sigma"));
    }

    Ok(())
}

pub fn validate_positive_size(size: Size) -> Result<()> {
    if size.width > 0 && size.height > 0 {
        return Ok(());
    }

    Err(Error::out_of_range("size"))
}

pub fn validate_source_offset(source: Size, source_offset: Point) -> Result<()> {
    if source_offset.x < 0 || source_offset.y < 0 {
        return Err(Error::out_of_range("source offset"));
    }
    if source_offset.x >= source.width || source_offset.y >= source.height {
        return Err(Error::out_of_range("source offset"));
    }

    Ok(())
}

pub fn validate_seed(size: Size, seed: Point) -> Result<()> {
    validate_positive_size(size)?;
    if seed.x >= 0 && seed.y >= 0 && seed.x < size.width && seed.y < size.height {
        return Ok(());
    }

    Err(Error::out_of_range("seed"))
}

pub fn validate_filter_mask(mask_size: Size, anchor: Point) -> Result<()> {
    if mask_size.width <= 0 || mask_size.height <= 0 {
        return Err(Error::out_of_range("mask size"));
    }
    if anchor.x < 0 || anchor.y < 0 || anchor.x >= mask_size.width || anchor.y >= mask_size.height {
        return Err(Error::out_of_range("anchor"));
    }

    Ok(())
}

pub fn validate_filter_kernel_2d<T>(kernel: &[T], kernel_size: Size, anchor: Point) -> Result<()> {
    validate_filter_mask(kernel_size, anchor)?;
    let width =
        usize::try_from(kernel_size.width).map_err(|_| Error::out_of_range("kernel width"))?;
    let height =
        usize::try_from(kernel_size.height).map_err(|_| Error::out_of_range("kernel height"))?;
    let expected = width
        .checked_mul(height)
        .ok_or_else(|| Error::out_of_range("kernel length"))?;
    if kernel.len() == expected {
        return Ok(());
    }

    Err(Error::out_of_range("kernel length"))
}

pub fn validate_border_roi(source: Size, source_offset: Point, roi: Size) -> Result<()> {
    if source_offset.x < 0 || source_offset.y < 0 {
        return Err(Error::out_of_range("source offset"));
    }

    let right = source_offset
        .x
        .checked_add(roi.width)
        .ok_or_else(|| Error::out_of_range("source offset"))?;
    let bottom = source_offset
        .y
        .checked_add(roi.height)
        .ok_or_else(|| Error::out_of_range("source offset"))?;

    if right <= source.width && bottom <= source.height {
        return Ok(());
    }

    Err(Error::SizeMismatch {
        name: "border roi".into(),
        expected: source,
        actual: roi,
    })
}

pub fn validate_canny_thresholds(low_threshold: i16, high_threshold: i16) -> Result<()> {
    if low_threshold < 0 || high_threshold < 0 {
        return Err(Error::out_of_range("threshold"));
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

pub fn validate_harris_corners_border_parameters(
    mask_size: MaskSize,
    average_window_size: MaskSize,
    border_type: BorderType,
) -> Result<()> {
    match mask_size {
        MaskSize::Mask3x3 | MaskSize::Mask5x5 => {}
        _ => {
            return Err(Error::out_of_range("mask size"));
        }
    }
    match average_window_size {
        MaskSize::Mask3x3 | MaskSize::Mask5x5 => {}
        _ => {
            return Err(Error::out_of_range("average window size"));
        }
    }
    if border_type == BorderType::Replicate {
        return Ok(());
    }
    Err(Error::out_of_range("border type"))
}

pub fn validate_hough_line_parameters(delta: PointPolar, max_line_count: i32) -> Result<()> {
    if !(delta.rho > 0.0 && delta.rho < 3.0) {
        return Err(Error::out_of_range("delta rho"));
    }
    if !(delta.theta > 0.25 && delta.theta < 3.0) {
        return Err(Error::out_of_range("delta theta"));
    }
    if max_line_count > 0 {
        return Ok(());
    }
    Err(Error::out_of_range("max line count"))
}

pub fn validate_hough_line_io(
    stream_context: &StreamContext,
    roi: Size,
    delta: PointPolar,
    threshold: i32,
    lines: &(impl DeviceOutput<PointPolar> + ?Sized),
    line_count: &(impl DeviceOutput<i32> + ?Sized),
) -> Result<()> {
    if lines.context() != stream_context.cuda_context()
        || line_count.context() != stream_context.cuda_context()
    {
        return Err(Error::StreamContextMismatch);
    }
    let max_line_count =
        i32::try_from(lines.len()).map_err(|_| Error::out_of_range("max line count"))?;
    validate_positive_size(roi)?;
    validate_hough_line_parameters(delta, max_line_count)?;
    if threshold <= 0 {
        return Err(Error::out_of_range("threshold"));
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

pub fn validate_hough_line_destination_roi(destination_roi: [PointPolar; 2]) -> Result<()> {
    if destination_roi[0].rho <= destination_roi[1].rho
        && destination_roi[0].theta <= destination_roi[1].theta
    {
        return Ok(());
    }
    Err(Error::out_of_range("destination roi"))
}

pub fn optional_image_output<T, L>(
    output: Option<&mut ImageOutputView<'_, T, L>>,
) -> (*mut T, i32, Option<Size>)
where
    L: ChannelLayout,
{
    output.map_or((ptr::null_mut(), 0, None), |image| {
        (image.as_mut_ptr().cast(), image.step(), Some(image.size()))
    })
}

pub fn validate_gradient_vector_outputs(
    destination_x: Option<Size>,
    destination_y: Option<Size>,
    destination_magnitude: Option<Size>,
    destination_angle: Option<Size>,
) -> Result<Size> {
    let roi = destination_x
        .or(destination_y)
        .or(destination_magnitude)
        .or(destination_angle)
        .ok_or_else(|| Error::out_of_range("gradient destination"))?;
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

pub fn validate_distance_transform_outputs(
    destination_voronoi: Option<Size>,
    destination_voronoi_indices: Option<Size>,
    destination_voronoi_manhattan: Option<Size>,
    destination_transform: Option<Size>,
) -> Result<Size> {
    let roi = destination_voronoi
        .or(destination_voronoi_indices)
        .or(destination_voronoi_manhattan)
        .or(destination_transform)
        .ok_or_else(|| Error::out_of_range("distance transform destination"))?;
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

pub fn validate_label_marker_step(source_destination: &ImageViewMut<'_, u32, C1>) -> Result<()> {
    validate_u32_label_marker_step(source_destination.size(), source_destination.step())
}

pub fn validate_u32_label_marker_step(size: Size, step: i32) -> Result<()> {
    let width =
        usize::try_from(size.width).map_err(|_| Error::out_of_range("source destination step"))?;
    let expected = width
        .checked_mul(size_of::<u32>())
        .ok_or_else(|| Error::out_of_range("source destination step"))?;
    let actual =
        usize::try_from(step).map_err(|_| Error::out_of_range("source destination step"))?;

    if actual == expected {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: "source destination step".into(),
        expected,
        actual,
    })
}

pub fn validate_marker_label_info_output(
    output: &DeviceMemory<CompressedMarkerLabelsInfo>,
    max_marker_label_id: u32,
) -> Result<()> {
    let expected = compressed_marker_labels_uf_info_list_size_u32_c1(max_marker_label_id)?
        .div_ceil(size_of::<CompressedMarkerLabelsInfo>());
    if output.len() >= expected {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: "marker labels info list".into(),
        expected,
        actual: output.len(),
    })
}

pub fn validate_marker_label_info_host_output(
    output: &[CompressedMarkerLabelsInfo],
    max_marker_label_id: u32,
) -> Result<()> {
    let expected = compressed_marker_labels_uf_info_list_size_u32_c1(max_marker_label_id)?
        .div_ceil(size_of::<CompressedMarkerLabelsInfo>());
    if output.len() >= expected {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: "marker labels info list".into(),
        expected,
        actual: output.len(),
    })
}

pub fn validate_u32_list_dev_output(
    output_dev: &DeviceMemory<u32>,
    max_marker_label_id: u32,
    name: &str,
) -> Result<()> {
    let expected = usize::try_from(max_marker_label_id)
        .map_err(|_| Error::out_of_range("max marker label id"))?;
    if output_dev.len() >= expected {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: name.to_owned().into(),
        expected,
        actual: output_dev.len(),
    })
}

pub fn validate_u32_list_host_output(output_host: &[u32], count: u32, name: &str) -> Result<()> {
    let expected = usize::try_from(count).map_err(|_| Error::out_of_range(name.to_owned()))?;
    if output_host.len() >= expected {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: name.to_owned().into(),
        expected,
        actual: output_host.len(),
    })
}

pub fn validate_u32_list_device_and_host_output(
    output_dev: &DeviceMemory<u32>,
    output_host: &[u32],
    count: u32,
    name: &str,
) -> Result<()> {
    validate_u32_list_dev_output(output_dev, count, name)?;
    validate_u32_list_host_output(output_host, count, name)
}

pub fn validate_u32_list_output(
    output_dev: &DeviceMemory<u32>,
    output_host: &[u32],
    max_marker_label_id: u32,
    name: &str,
) -> Result<()> {
    let expected = usize::try_from(max_marker_label_id)
        .map_err(|_| Error::out_of_range("max marker label id"))?;
    if output_dev.len() < expected {
        return Err(Error::LengthMismatch {
            name: name.to_owned().into(),
            expected,
            actual: output_dev.len(),
        });
    }
    if output_host.len() < expected {
        return Err(Error::LengthMismatch {
            name: name.to_owned().into(),
            expected,
            actual: output_host.len(),
        });
    }
    Ok(())
}

pub fn validate_typed_output_len(actual: usize, expected_count: u32, name: &str) -> Result<()> {
    let expected =
        usize::try_from(expected_count).map_err(|_| Error::out_of_range(name.to_owned()))?;
    if actual == expected {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: name.to_owned().into(),
        expected,
        actual,
    })
}

pub fn validate_geometry_list_id_range(
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

    Err(Error::out_of_range("last contour geometry list id"))
}

pub fn validate_geometry_list_buffers(
    output_dev: &DeviceMemory<ContourPixelGeometryInfo>,
    output_host: &[ContourPixelGeometryInfo],
    max_contour_pixel_geometry_info_count: u32,
) -> Result<()> {
    let required =
        compressed_marker_labels_uf_geometry_lists_size_c1(max_contour_pixel_geometry_info_count)?
            .div_ceil(size_of::<ContourPixelGeometryInfo>());
    if output_dev.len() < required || output_host.len() < required {
        return Err(Error::LengthMismatch {
            name: "contours pixel geometry lists".into(),
            expected: required,
            actual: output_dev.len().min(output_host.len()),
        });
    }
    if output_dev.len() == output_host.len() {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: "contours pixel geometry lists".into(),
        expected: output_host.len(),
        actual: output_dev.len(),
    })
}

pub fn validate_contours_geometry_image_host(output: &[u8], size: Size, step: i32) -> Result<()> {
    let step =
        usize::try_from(step).map_err(|_| Error::out_of_range("contours geometry image step"))?;
    let width = usize::try_from(size.width)
        .map_err(|_| Error::out_of_range("contours geometry image width"))?;
    let height = usize::try_from(size.height)
        .map_err(|_| Error::out_of_range("contours geometry image height"))?;
    let required = if height == 0 {
        0
    } else {
        step.checked_mul(height - 1)
            .and_then(|bytes| bytes.checked_add(width))
            .ok_or_else(|| Error::out_of_range("contours geometry image"))?
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

pub fn validate_block_segment_inputs(
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

pub fn validate_equal_size(images: &[ImageViewMut<'_, u32, C1>]) -> Result<Size> {
    let Some(first) = images.first() else {
        return Err(Error::out_of_range("batch size"));
    };
    let size = first.size();
    for image in &images[1..] {
        validate_same_size(size, image.size())?;
    }
    Ok(size)
}

pub fn label_markers_uf_batch_descriptors<T>(
    allocation_context: &Arc<CudaContext>,
    sources: &[ImageView<'_, T, C1>],
    destinations: &mut [ImageOutputView<'_, u32, C1>],
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
            .chain(destinations.iter().map(ImageOutputView::size)),
    )?;
    let source_descriptors = image_descriptors(sources);
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        roi,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &source_descriptors) }?,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(allocation_context, &destination_descriptors)
        }?,
    ))
}

pub fn label_markers_uf_batch_advanced_descriptors<T>(
    allocation_context: &Arc<CudaContext>,
    sources: &[ImageView<'_, T, C1>],
    destinations: &mut [ImageOutputView<'_, u32, C1>],
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

    let max_roi = largest_size(destinations.iter().map(ImageOutputView::size))?;
    let source_descriptors = image_descriptors(sources);
    let destination_descriptors = image_descriptors_mut(destinations);

    Ok((
        batch_size,
        max_roi,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &source_descriptors) }?,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe {
            DeviceMemory::from_slice_unchecked(allocation_context, &destination_descriptors)
        }?,
    ))
}

pub fn compress_marker_labels_uf_batch_descriptors(
    allocation_context: &Arc<CudaContext>,
    source_destinations: &mut [ImageViewMut<'_, u32, C1>],
) -> Result<(i32, Size, DeviceMemory<sys::NppiImageDescriptor>)> {
    let batch_size =
        label_markers_uf_batch_size(source_destinations.len(), source_destinations.len())?;
    let roi = validate_equal_size(source_destinations)?;
    for image in source_destinations.iter() {
        validate_label_marker_step(image)?;
    }

    let descriptors = image_descriptors_mut(source_destinations);

    // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
    Ok((batch_size, roi, unsafe {
        DeviceMemory::from_slice_unchecked(allocation_context, &descriptors)
    }?))
}

pub fn compress_marker_labels_uf_batch_advanced_descriptors(
    allocation_context: &Arc<CudaContext>,
    source_destinations: &mut [ImageViewMut<'_, u32, C1>],
) -> Result<(i32, Size, DeviceMemory<sys::NppiImageDescriptor>)> {
    let batch_size =
        label_markers_uf_batch_size(source_destinations.len(), source_destinations.len())?;
    for image in source_destinations.iter() {
        validate_label_marker_step(image)?;
    }

    let max_roi = largest_size(source_destinations.iter().map(ImageViewMut::size))?;
    let descriptors = image_descriptors_mut(source_destinations);

    // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
    Ok((batch_size, max_roi, unsafe {
        DeviceMemory::from_slice_unchecked(allocation_context, &descriptors)
    }?))
}

pub fn compress_marker_labels_uf_batch_buffers(
    allocation_context: &Arc<CudaContext>,
    batch_size: usize,
    bytes_per_image: usize,
) -> Result<(Vec<DeviceMemory<MaybeUninit<u8>>>, BufferDescriptors)> {
    create_repeated_buffer_descriptors(allocation_context, batch_size, bytes_per_image)
}

pub fn compress_marker_labels_uf_batch_advanced_buffers(
    allocation_context: &Arc<CudaContext>,
    source_destinations: &[ImageViewMut<'_, u32, C1>],
) -> Result<AdvancedMarkerBuffers> {
    let mut largest_per_image_bytes = 0_usize;
    let mut buffer_sizes = Vec::with_capacity(source_destinations.len());

    for image in source_destinations {
        let per_image_bytes =
            compress_marker_labels_uf_buffer_size_u32_c1(roi_area(image.size())?)?;
        largest_per_image_bytes = largest_per_image_bytes.max(per_image_bytes);
        buffer_sizes.push(per_image_bytes);
    }
    let (buffers, descriptors) = create_buffer_descriptors(allocation_context, buffer_sizes)?;

    Ok((largest_per_image_bytes, buffers, descriptors))
}

pub fn validate_batch_label_output(
    output: &(impl DeviceOutput<u32> + ?Sized),
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

pub fn roi_area(roi: Size) -> Result<i32> {
    roi.width
        .checked_mul(roi.height)
        .ok_or_else(|| Error::out_of_range("starting number"))
}

pub fn label_markers_uf_batch_size(source_len: usize, destination_len: usize) -> Result<i32> {
    batch_size_i32(source_len, destination_len)
}

pub fn smallest_size(sizes: impl IntoIterator<Item = Size>) -> Result<Size> {
    sizes
        .into_iter()
        .reduce(|left, right| Size {
            width: left.width.min(right.width),
            height: left.height.min(right.height),
        })
        .ok_or_else(|| Error::out_of_range("batch size"))
}

pub fn largest_size(sizes: impl IntoIterator<Item = Size>) -> Result<Size> {
    sizes
        .into_iter()
        .reduce(|left, right| Size {
            width: left.width.max(right.width),
            height: left.height.max(right.height),
        })
        .ok_or_else(|| Error::out_of_range("batch size"))
}

pub type AdvancedMarkerBuffers = (usize, Vec<DeviceMemory<MaybeUninit<u8>>>, BufferDescriptors);
