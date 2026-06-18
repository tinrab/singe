use singe_cuda::memory::DeviceMemory;

use crate::error::{Error, Result};

use super::{ImageHistograms, ImageStatistic};

pub(crate) fn histogram_buffers<const C: usize>(
    levels: [i32; C],
) -> Result<[DeviceMemory<i32>; C]> {
    let lengths = levels.map(histogram_bins);
    let lengths = transpose_result(lengths)?;
    let mut buffers = Vec::with_capacity(C);
    for length in lengths {
        buffers.push(DeviceMemory::<i32>::create(length)?);
    }
    match buffers.try_into() {
        Ok(buffers) => Ok(buffers),
        Err(buffers) => Err(Error::LengthMismatch {
            name: "array length".into(),
            expected: C,
            actual: buffers.len(),
        }),
    }
}

pub(crate) fn histogram_range_buffers<Level, const C: usize>(
    levels: [&[Level]; C],
) -> Result<[DeviceMemory<i32>; C]> {
    let lengths = levels.map(histogram_range_bins);
    let lengths = transpose_result(lengths)?;
    let mut buffers = Vec::with_capacity(C);
    for length in lengths {
        buffers.push(DeviceMemory::<i32>::create(length)?);
    }
    match buffers.try_into() {
        Ok(buffers) => Ok(buffers),
        Err(buffers) => Err(Error::LengthMismatch {
            name: "array length".into(),
            expected: C,
            actual: buffers.len(),
        }),
    }
}

fn histogram_bins(levels: i32) -> Result<usize> {
    if levels < 2 {
        return Err(Error::OutOfRange {
            name: "levels".into(),
        });
    }
    usize::try_from(levels - 1).map_err(|_| Error::OutOfRange {
        name: "levels".into(),
    })
}

fn histogram_range_bins<Level>(levels: &[Level]) -> Result<usize> {
    if levels.len() < 2 {
        return Err(Error::OutOfRange {
            name: "levels".into(),
        });
    }
    Ok(levels.len() - 1)
}

fn transpose_result<T, const C: usize>(results: [Result<T>; C]) -> Result<[T; C]> {
    let mut values = Vec::with_capacity(C);
    for result in results {
        values.push(result?);
    }
    match values.try_into() {
        Ok(values) => Ok(values),
        Err(values) => Err(Error::LengthMismatch {
            name: "array length".into(),
            expected: C,
            actual: values.len(),
        }),
    }
}

pub(crate) fn image_histograms(histograms: Vec<DeviceMemory<i32>>) -> ImageHistograms {
    ImageHistograms {
        channels: histograms
            .into_iter()
            .map(ImageStatistic::from_values)
            .collect(),
    }
}
