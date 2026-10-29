use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::{
    context::Context as CudaContext,
    memory::DeviceMemory,
    view::{DeviceOutput, DeviceOutputView, DeviceRepr},
};
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{ChannelLayout, ImageView, image_descriptors},
    types::{Point, ProfileStatistics, Rectangle, Size},
    utility::batch_size_i32,
    workspace::{BufferDescriptors, create_repeated_buffer_descriptors},
};

pub fn pair_metric_batch_descriptors<T, L>(
    allocation_context: &Arc<CudaContext>,
    source_0: &[ImageView<'_, T, L>],
    source_1: &[ImageView<'_, T, L>],
) -> Result<(
    i32,
    Size,
    DeviceMemory<sys::NppiImageDescriptor>,
    DeviceMemory<sys::NppiImageDescriptor>,
)>
where
    L: ChannelLayout,
{
    let batch_size = pair_batch_size(source_0.len(), source_1.len())?;

    for (left, right) in source_0.iter().zip(source_1.iter()) {
        validate_same_size(left.size(), right.size())?;
    }

    let roi = smallest_size(
        source_0
            .iter()
            .map(ImageView::size)
            .chain(source_1.iter().map(ImageView::size)),
    )?;
    let descriptors_0 = image_descriptors(source_0);
    let descriptors_1 = image_descriptors(source_1);

    Ok((
        batch_size,
        roi,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &descriptors_0) }?,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &descriptors_1) }?,
    ))
}

pub fn pair_metric_batch_advanced_descriptors<T, L>(
    allocation_context: &Arc<CudaContext>,
    source_0: &[ImageView<'_, T, L>],
    source_1: &[ImageView<'_, T, L>],
) -> Result<(
    i32,
    Size,
    DeviceMemory<sys::NppiImageDescriptor>,
    DeviceMemory<sys::NppiImageDescriptor>,
)>
where
    L: ChannelLayout,
{
    let batch_size = pair_batch_size(source_0.len(), source_1.len())?;

    for (left, right) in source_0.iter().zip(source_1.iter()) {
        validate_same_size(left.size(), right.size())?;
    }

    let max_roi = largest_size(
        source_0
            .iter()
            .map(ImageView::size)
            .chain(source_1.iter().map(ImageView::size)),
    )?;
    let descriptors_0 = image_descriptors(source_0);
    let descriptors_1 = image_descriptors(source_1);

    Ok((
        batch_size,
        max_roi,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &descriptors_0) }?,
        // SAFETY: the descriptors are plain C structs, so copying them yields valid device values.
        unsafe { DeviceMemory::from_slice_unchecked(allocation_context, &descriptors_1) }?,
    ))
}

pub fn pair_metric_batch_buffers(
    allocation_context: &Arc<CudaContext>,
    batch_size: usize,
    bytes_per_image: usize,
) -> Result<(Vec<DeviceMemory<MaybeUninit<u8>>>, BufferDescriptors)> {
    create_repeated_buffer_descriptors(allocation_context, batch_size, bytes_per_image)
}

pub fn pair_batch_size(source_0_len: usize, source_1_len: usize) -> Result<i32> {
    batch_size_i32(source_0_len, source_1_len)
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

pub fn validate_template_full_destination_size(
    source: Size,
    template: Size,
    destination: Size,
) -> Result<()> {
    let expected = Size {
        width: source
            .width
            .checked_add(template.width)
            .and_then(|width| width.checked_sub(1))
            .ok_or_else(|| Error::out_of_range("template full destination width"))?,
        height: source
            .height
            .checked_add(template.height)
            .and_then(|height| height.checked_sub(1))
            .ok_or_else(|| Error::out_of_range("template full destination height"))?,
    };

    validate_same_size(expected, destination)
}

pub fn validate_template_valid_destination_size(
    source: Size,
    template: Size,
    destination: Size,
) -> Result<()> {
    let expected = Size {
        width: source
            .width
            .checked_sub(template.width)
            .and_then(|width| width.checked_add(1))
            .ok_or_else(|| Error::out_of_range("template valid destination width"))?,
        height: source
            .height
            .checked_sub(template.height)
            .and_then(|height| height.checked_add(1))
            .ok_or_else(|| Error::out_of_range("template valid destination height"))?,
    };

    validate_same_size(expected, destination)
}

pub fn validate_integral_destination_size(source: Size, destination: Size) -> Result<()> {
    let expected = Size {
        width: source
            .width
            .checked_add(1)
            .ok_or_else(|| Error::out_of_range("integral destination width"))?,
        height: source
            .height
            .checked_add(1)
            .ok_or_else(|| Error::out_of_range("integral destination height"))?,
    };

    if expected == destination {
        return Ok(());
    }

    Err(Error::SizeMismatch {
        name: "integral destination size".into(),
        expected,
        actual: destination,
    })
}

pub fn validate_rect_std_dev_sources(
    source: Size,
    squared: Size,
    destination: Size,
    rectangle: Rectangle,
) -> Result<()> {
    if rectangle.x < 0 || rectangle.y < 0 || rectangle.width <= 0 || rectangle.height <= 0 {
        return Err(Error::out_of_range("rectangle"));
    }

    let expected = Size {
        width: destination
            .width
            .checked_add(rectangle.x)
            .and_then(|value| value.checked_add(rectangle.width))
            .ok_or_else(|| Error::out_of_range("rect stddev source width"))?,
        height: destination
            .height
            .checked_add(rectangle.y)
            .and_then(|value| value.checked_add(rectangle.height))
            .ok_or_else(|| Error::out_of_range("rect stddev source height"))?,
    };

    validate_same_size(expected, source)?;
    validate_same_size(expected, squared)?;
    Ok(())
}

pub fn validate_metric_output(
    output: &(impl DeviceOutput<f32> + ?Sized),
    batch_size: usize,
    channels: usize,
) -> Result<()> {
    let expected = batch_size
        .checked_mul(channels)
        .ok_or_else(|| Error::out_of_range("metric output length"))?;
    if output.len() != expected {
        return Err(Error::LengthMismatch {
            name: "metric output length".into(),
            expected,
            actual: output.len(),
        });
    }
    Ok(())
}

pub fn validate_statistic_output(
    output: &DeviceOutputView<'_, f64>,
    channels: usize,
) -> Result<()> {
    if output.len() != channels {
        return Err(Error::LengthMismatch {
            name: "statistic output length".into(),
            expected: channels,
            actual: output.len(),
        });
    }
    Ok(())
}

pub fn validate_typed_statistic_output<T: DeviceRepr>(
    output: &DeviceOutputView<'_, T>,
    channels: usize,
) -> Result<()> {
    if output.len() != channels {
        return Err(Error::LengthMismatch {
            name: "statistic output length".into(),
            expected: channels,
            actual: output.len(),
        });
    }
    Ok(())
}

pub fn validate_profile_output(output: &[ProfileStatistics]) -> Result<u16> {
    if output.is_empty() {
        return Err(Error::LengthMismatch {
            name: "profile output length".into(),
            expected: 1,
            actual: 0,
        });
    }

    u16::try_from(output.len()).map_err(|_| Error::out_of_range("profile output length"))
}

pub fn validate_index_output(output: &DeviceOutputView<'_, i32>, channels: usize) -> Result<()> {
    if output.len() != channels {
        return Err(Error::LengthMismatch {
            name: "index output length".into(),
            expected: channels,
            actual: output.len(),
        });
    }
    Ok(())
}

pub fn validate_point_output(output: &DeviceOutputView<'_, Point>, channels: usize) -> Result<()> {
    if output.len() != channels {
        return Err(Error::LengthMismatch {
            name: "point output length".into(),
            expected: channels,
            actual: output.len(),
        });
    }
    Ok(())
}

pub fn validate_mask_size(roi: Size, mask: Size) -> Result<()> {
    if roi != mask {
        return Err(Error::SizeMismatch {
            name: "mask size".into(),
            expected: roi,
            actual: mask,
        });
    }
    Ok(())
}

pub fn validate_histogram_levels(levels: i32) -> Result<usize> {
    if levels < 2 {
        return Err(Error::out_of_range("levels"));
    }
    usize::try_from(levels - 1).map_err(|_| Error::out_of_range("levels"))
}

pub fn validate_histogram_levels_array<const C: usize>(levels: &[i32; C]) -> Result<()> {
    for &level in levels {
        validate_histogram_levels(level)?;
    }
    Ok(())
}

pub fn validate_histogram_output(
    stream_context: &StreamContext,
    histogram: &(impl DeviceOutput<i32> + ?Sized),
    levels: i32,
) -> Result<()> {
    if histogram.context() != stream_context.cuda_context() {
        return Err(Error::StreamContextMismatch);
    }
    let expected_bins = validate_histogram_levels(levels)?;
    if histogram.len() != expected_bins {
        return Err(Error::LengthMismatch {
            name: "histogram".into(),
            expected: expected_bins,
            actual: histogram.len(),
        });
    }
    Ok(())
}

pub fn validate_histogram_outputs<const C: usize>(
    stream_context: &StreamContext,
    histograms: &[&mut (impl DeviceOutput<i32> + ?Sized); C],
    levels: &[i32; C],
) -> Result<()> {
    for histogram in histograms {
        if histogram.context() != stream_context.cuda_context() {
            return Err(Error::StreamContextMismatch);
        }
    }
    for (index, (&level, histogram)) in levels.iter().zip(histograms.iter()).enumerate() {
        let expected_bins = validate_histogram_levels(level)?;
        if histogram.len() != expected_bins {
            return Err(Error::LengthMismatch {
                name: format!("histogram[{index}]").into(),
                expected: expected_bins,
                actual: histogram.len(),
            });
        }
    }
    Ok(())
}

pub fn validate_histogram_range_levels<T>(levels: &[T]) -> Result<i32> {
    if levels.len() < 2 {
        return Err(Error::out_of_range("levels"));
    }
    i32::try_from(levels.len()).map_err(|_| Error::out_of_range("levels"))
}

pub fn validate_histogram_range_level_arrays<T, const C: usize>(
    levels: &[&[T]; C],
) -> Result<[i32; C]> {
    let mut raw_levels = [0_i32; C];
    for (index, level) in levels.iter().enumerate() {
        raw_levels[index] = validate_histogram_range_levels(level)?;
    }
    Ok(raw_levels)
}

pub fn validate_histogram_range_output<T>(
    stream_context: &StreamContext,
    histogram: &(impl DeviceOutput<i32> + ?Sized),
    levels: &[T],
) -> Result<i32> {
    if histogram.context() != stream_context.cuda_context() {
        return Err(Error::StreamContextMismatch);
    }
    let raw_levels = validate_histogram_range_levels(levels)?;
    let expected_bins = levels.len() - 1;
    if histogram.len() != expected_bins {
        return Err(Error::LengthMismatch {
            name: "histogram".into(),
            expected: expected_bins,
            actual: histogram.len(),
        });
    }
    Ok(raw_levels)
}

pub fn validate_histogram_range_outputs<T, const C: usize>(
    stream_context: &StreamContext,
    histograms: &[&mut (impl DeviceOutput<i32> + ?Sized); C],
    levels: &[&[T]; C],
) -> Result<[i32; C]> {
    for histogram in histograms {
        if histogram.context() != stream_context.cuda_context() {
            return Err(Error::StreamContextMismatch);
        }
    }
    let raw_levels = validate_histogram_range_level_arrays(levels)?;
    for (index, (histogram, level)) in histograms.iter().zip(levels.iter()).enumerate() {
        let expected_bins = level.len() - 1;
        if histogram.len() != expected_bins {
            return Err(Error::LengthMismatch {
                name: format!("histogram[{index}]").into(),
                expected: expected_bins,
                actual: histogram.len(),
            });
        }
    }
    Ok(raw_levels)
}

pub fn validate_coi(channel: usize, channels: usize) -> Result<i32> {
    if channel < channels {
        return i32::try_from(channel + 1).map_err(|_| Error::out_of_range("channel"));
    }

    Err(Error::out_of_range("channel"))
}
