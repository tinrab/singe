//! Softmax and log-sum-exp style kernels.

use std::sync::Arc;

#[cfg(feature = "cutile")]
use ::cutile::cuda_async::device_buffer::DevicePointer;
#[cfg(feature = "dtype-f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceRepr, DeviceSlice, DeviceSliceMut},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::{borrowed_stream, input_pointer, output_pointer},
    error::{Error, Result},
    utility::{checked_element_count, ensure_len},
};

macro_rules! softmax_fns {
    ($ty:ty, $softmax:ident, $log_softmax:ident, $softmin:ident) => {
        pub fn $softmax(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            softmax(stream, out, input, rows, cols, cutile::softmax::$softmax)
        }

        pub fn $log_softmax(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            softmax(
                stream,
                out,
                input,
                rows,
                cols,
                cutile::softmax::$log_softmax,
            )
        }

        pub fn $softmin(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            softmax(stream, out, input, rows, cols, cutile::softmax::$softmin)
        }
    };
}

#[cfg(feature = "dtype-f32")]
softmax_fns!(f32, softmax_f32, log_softmax_f32, softmin_f32);
#[cfg(feature = "dtype-f16")]
softmax_fns!(f16, softmax_f16, log_softmax_f16, softmin_f16);
#[cfg(feature = "dtype-f64")]
softmax_fns!(f64, softmax_f64, log_softmax_f64, softmin_f64);

#[cfg(feature = "dtype-f32")]
pub fn softmax_lse_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_lse(
        stream,
        out,
        input,
        rows,
        cols,
        cutile::softmax::softmax_lse_f32,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn softmax_lse_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f16>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_lse(
        stream,
        out,
        input,
        rows,
        cols,
        cutile::softmax::softmax_lse_f16,
    )
}

#[cfg(feature = "dtype-f64")]
pub fn softmax_lse_f64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f64>,
    input: &impl DeviceSlice<f64>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_lse(
        stream,
        out,
        input,
        rows,
        cols,
        cutile::softmax::softmax_lse_f64,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn softmax_lse_wide_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    partial: &mut impl DeviceSliceMut<f32>,
    row_max: &mut impl DeviceSliceMut<f32>,
    row_sum: &mut impl DeviceSliceMut<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_lse_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        cutile::softmax::softmax_lse_wide_f32,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn softmax_lse_wide_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f16>,
    partial: &mut impl DeviceSliceMut<f32>,
    row_max: &mut impl DeviceSliceMut<f32>,
    row_sum: &mut impl DeviceSliceMut<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_lse_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        cutile::softmax::softmax_lse_wide_f16,
    )
}

#[cfg(feature = "dtype-f64")]
pub fn softmax_lse_wide_f64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f64>,
    input: &impl DeviceSlice<f64>,
    partial: &mut impl DeviceSliceMut<f64>,
    row_max: &mut impl DeviceSliceMut<f64>,
    row_sum: &mut impl DeviceSliceMut<f64>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_lse_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        cutile::softmax::softmax_lse_wide_f64,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn softmax_wide_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    partial: &mut impl DeviceSliceMut<f32>,
    row_max: &mut impl DeviceSliceMut<f32>,
    row_sum: &mut impl DeviceSliceMut<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        cutile::softmax::softmax_wide_f32,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn softmax_wide_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    input: &impl DeviceSlice<f16>,
    partial: &mut impl DeviceSliceMut<f32>,
    row_max: &mut impl DeviceSliceMut<f32>,
    row_sum: &mut impl DeviceSliceMut<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        cutile::softmax::softmax_wide_f16,
    )
}

#[cfg(feature = "dtype-f64")]
pub fn softmax_wide_f64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f64>,
    input: &impl DeviceSlice<f64>,
    partial: &mut impl DeviceSliceMut<f64>,
    row_max: &mut impl DeviceSliceMut<f64>,
    row_sum: &mut impl DeviceSliceMut<f64>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        cutile::softmax::softmax_wide_f64,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn log_softmax_wide_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    partial: &mut impl DeviceSliceMut<f32>,
    row_max: &mut impl DeviceSliceMut<f32>,
    row_sum: &mut impl DeviceSliceMut<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        cutile::softmax::log_softmax_wide_f32,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn log_softmax_wide_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    input: &impl DeviceSlice<f16>,
    partial: &mut impl DeviceSliceMut<f32>,
    row_max: &mut impl DeviceSliceMut<f32>,
    row_sum: &mut impl DeviceSliceMut<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        cutile::softmax::log_softmax_wide_f16,
    )
}

#[cfg(feature = "dtype-f64")]
pub fn log_softmax_wide_f64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f64>,
    input: &impl DeviceSlice<f64>,
    partial: &mut impl DeviceSliceMut<f64>,
    row_max: &mut impl DeviceSliceMut<f64>,
    row_sum: &mut impl DeviceSliceMut<f64>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_max,
        row_sum,
        rows,
        cols,
        cutile::softmax::log_softmax_wide_f64,
    )
}

#[cfg(feature = "dtype-f32")]
pub fn softmin_wide_f32(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f32>,
    input: &impl DeviceSlice<f32>,
    partial: &mut impl DeviceSliceMut<f32>,
    row_min: &mut impl DeviceSliceMut<f32>,
    row_sum: &mut impl DeviceSliceMut<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_min,
        row_sum,
        rows,
        cols,
        cutile::softmax::softmin_wide_f32,
    )
}

#[cfg(feature = "dtype-f16")]
pub fn softmin_wide_f16(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f16>,
    input: &impl DeviceSlice<f16>,
    partial: &mut impl DeviceSliceMut<f32>,
    row_min: &mut impl DeviceSliceMut<f32>,
    row_sum: &mut impl DeviceSliceMut<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_min,
        row_sum,
        rows,
        cols,
        cutile::softmax::softmin_wide_f16,
    )
}

#[cfg(feature = "dtype-f64")]
pub fn softmin_wide_f64(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<f64>,
    input: &impl DeviceSlice<f64>,
    partial: &mut impl DeviceSliceMut<f64>,
    row_min: &mut impl DeviceSliceMut<f64>,
    row_sum: &mut impl DeviceSliceMut<f64>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    softmax_wide(
        stream,
        out,
        input,
        partial,
        row_min,
        row_sum,
        rows,
        cols,
        cutile::softmax::softmin_wide_f64,
    )
}

fn softmax<T, F>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<T>,
    input: &impl DeviceSlice<T>,
    rows: usize,
    cols: usize,
    launch: F,
) -> Result<()>
where
    T: DeviceRepr,
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<T>,
        DevicePointer<T>,
        usize,
        usize,
    ) -> Result<()>,
{
    if cols == 0 {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), len)?;
    ensure_len(input.len(), len)?;
    let stream = borrowed_stream(stream)?;
    launch(
        &stream,
        output_pointer(out),
        input_pointer(input),
        rows,
        cols,
    )
}

fn softmax_wide<Input, Scratch, F>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<Input>,
    input: &impl DeviceSlice<Input>,
    partial: &mut impl DeviceSliceMut<Scratch>,
    row_max: &mut impl DeviceSliceMut<Scratch>,
    row_sum: &mut impl DeviceSliceMut<Scratch>,
    rows: usize,
    cols: usize,
    launch: F,
) -> Result<()>
where
    Input: DeviceRepr,
    Scratch: DeviceRepr,
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<Input>,
        DevicePointer<Input>,
        DevicePointer<Scratch>,
        DevicePointer<Scratch>,
        DevicePointer<Scratch>,
        usize,
        usize,
    ) -> Result<()>,
{
    if cols == 0 {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), len)?;
    ensure_len(input.len(), len)?;
    let chunks = cols.div_ceil(1024);
    let partial_len = checked_element_count(rows, chunks)?;
    ensure_len(partial.len(), partial_len)?;
    ensure_len(row_max.len(), rows)?;
    ensure_len(row_sum.len(), rows)?;
    let stream = borrowed_stream(stream)?;
    launch(
        &stream,
        output_pointer(out),
        input_pointer(input),
        output_pointer(partial),
        output_pointer(row_max),
        output_pointer(row_sum),
        rows,
        cols,
    )
}

fn softmax_lse<Input, Output, F>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<Output>,
    input: &impl DeviceSlice<Input>,
    rows: usize,
    cols: usize,
    launch: F,
) -> Result<()>
where
    Input: DeviceRepr,
    Output: DeviceRepr,
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<Output>,
        DevicePointer<Input>,
        usize,
        usize,
    ) -> Result<()>,
{
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), rows)?;
    ensure_len(input.len(), len)?;
    let stream = borrowed_stream(stream)?;
    launch(
        &stream,
        output_pointer(out),
        input_pointer(input),
        rows,
        cols,
    )
}

fn softmax_lse_wide<Input, Output, Scratch, F>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<Output>,
    input: &impl DeviceSlice<Input>,
    partial: &mut impl DeviceSliceMut<Scratch>,
    row_max: &mut impl DeviceSliceMut<Scratch>,
    row_sum: &mut impl DeviceSliceMut<Scratch>,
    rows: usize,
    cols: usize,
    launch: F,
) -> Result<()>
where
    Input: DeviceRepr,
    Output: DeviceRepr,
    Scratch: DeviceRepr,
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<Output>,
        DevicePointer<Input>,
        DevicePointer<Scratch>,
        DevicePointer<Scratch>,
        DevicePointer<Scratch>,
        usize,
        usize,
    ) -> Result<()>,
{
    if cols == 0 {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), rows)?;
    ensure_len(input.len(), len)?;
    let chunks = cols.div_ceil(1024);
    let partial_len = checked_element_count(rows, chunks)?;
    ensure_len(partial.len(), partial_len)?;
    ensure_len(row_max.len(), rows)?;
    ensure_len(row_sum.len(), rows)?;
    let stream = borrowed_stream(stream)?;
    launch(
        &stream,
        output_pointer(out),
        input_pointer(input),
        output_pointer(partial),
        output_pointer(row_max),
        output_pointer(row_sum),
        rows,
        cols,
    )
}
