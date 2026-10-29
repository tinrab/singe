use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory};

use crate::{
    context::StreamContext,
    error::{Error, Result},
    pipeline::image::statistics_histogram_channel_methods::{ImageHistograms, ImageStatistic},
};

pub fn histogram_buffers<const C: usize>(
    context: &Arc<CudaContext>,
    levels: [i32; C],
) -> Result<[DeviceMemory<MaybeUninit<i32>>; C]> {
    let lengths = levels.map(histogram_bins);
    let lengths = transpose_result(lengths)?;
    let mut buffers = Vec::with_capacity(C);
    for length in lengths {
        buffers.push(DeviceMemory::<i32>::create_uninit(context, length)?);
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

pub fn histogram_range_buffers<Level, const C: usize>(
    context: &Arc<CudaContext>,
    levels: [&[Level]; C],
) -> Result<[DeviceMemory<MaybeUninit<i32>>; C]> {
    let lengths = levels.map(histogram_range_bins);
    let lengths = transpose_result(lengths)?;
    let mut buffers = Vec::with_capacity(C);
    for length in lengths {
        buffers.push(DeviceMemory::<i32>::create_uninit(context, length)?);
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
        return Err(Error::out_of_range("levels"));
    }
    usize::try_from(levels - 1).map_err(|_| Error::out_of_range("levels"))
}

fn histogram_range_bins<Level>(levels: &[Level]) -> Result<usize> {
    if levels.len() < 2 {
        return Err(Error::out_of_range("levels"));
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

/// Every bin must have been written by a producer already queued on `stream_context`.
pub unsafe fn image_histograms(
    stream_context: &StreamContext,
    histograms: Vec<DeviceMemory<MaybeUninit<i32>>>,
) -> Result<ImageHistograms> {
    stream_context.synchronize()?;
    Ok(ImageHistograms {
        channels: histograms
            .into_iter()
            // SAFETY: the caller upholds this function's safety contract.
            .map(|histogram| ImageStatistic::from_values(unsafe { histogram.assume_init() }))
            .collect(),
    })
}
