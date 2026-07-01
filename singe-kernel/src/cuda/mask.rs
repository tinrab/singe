//! Boolean masks, causal masks, and mask composition.

#[cfg(feature = "cutile")]
use std::sync::Arc;

#[cfg(feature = "dtype-bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype-f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceSlice, DeviceSliceMut},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::{borrowed_stream, input_pointer, output_pointer},
    error::{Error, Result},
    utility::{checked_element_count, ensure_len},
};
#[cfg(feature = "cutile")]
use ::cutile::cuda_async::device_buffer::DevicePointer;

pub fn mask_not_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    input: &impl DeviceSlice<u8>,
) -> Result<()> {
    let len = out.len();
    ensure_len(input.len(), len)?;
    let stream = borrowed_stream(stream)?;
    cutile::mask::mask_not_u8(&stream, output_pointer(out), input_pointer(input), len)
}

pub fn mask_and_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    lhs: &impl DeviceSlice<u8>,
    rhs: &impl DeviceSlice<u8>,
) -> Result<()> {
    mask_binary(stream, out, lhs, rhs, cutile::mask::mask_and_u8)
}

pub fn mask_or_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    lhs: &impl DeviceSlice<u8>,
    rhs: &impl DeviceSlice<u8>,
) -> Result<()> {
    mask_binary(stream, out, lhs, rhs, cutile::mask::mask_or_u8)
}

pub fn mask_xor_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    lhs: &impl DeviceSlice<u8>,
    rhs: &impl DeviceSlice<u8>,
) -> Result<()> {
    mask_binary(stream, out, lhs, rhs, cutile::mask::mask_xor_u8)
}

pub fn logical_not_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    input: &impl DeviceSlice<u8>,
) -> Result<()> {
    mask_not_u8(stream, out, input)
}

pub fn logical_and_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    lhs: &impl DeviceSlice<u8>,
    rhs: &impl DeviceSlice<u8>,
) -> Result<()> {
    mask_and_u8(stream, out, lhs, rhs)
}

pub fn logical_or_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    lhs: &impl DeviceSlice<u8>,
    rhs: &impl DeviceSlice<u8>,
) -> Result<()> {
    mask_or_u8(stream, out, lhs, rhs)
}

pub fn logical_xor_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    lhs: &impl DeviceSlice<u8>,
    rhs: &impl DeviceSlice<u8>,
) -> Result<()> {
    mask_xor_u8(stream, out, lhs, rhs)
}

pub fn mask_any_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    input: &impl DeviceSlice<u8>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    mask_reduction(stream, out, input, rows, cols, cutile::mask::mask_any_u8)
}

pub fn mask_all_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    input: &impl DeviceSlice<u8>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    mask_reduction(stream, out, input, rows, cols, cutile::mask::mask_all_u8)
}

pub fn mask_count_nonzero_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u32>,
    input: &impl DeviceSlice<u8>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    if cols == 0 {
        return Err(Error::InvalidLength);
    }
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), rows)?;
    ensure_len(input.len(), len)?;
    let stream = borrowed_stream(stream)?;
    cutile::mask::mask_count_nonzero_u8(
        &stream,
        output_pointer(out),
        input_pointer(input),
        rows,
        cols,
    )
}

pub fn tril_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    triangular_mask(
        stream,
        out,
        rows,
        cols,
        diagonal,
        cutile::mask::tril_mask_u8,
    )
}

pub fn triu_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    triangular_mask(
        stream,
        out,
        rows,
        cols,
        diagonal,
        cutile::mask::triu_mask_u8,
    )
}

pub fn lower_triangular_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    tril_mask_u8(stream, out, rows, cols, diagonal)
}

pub fn upper_triangular_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    triu_mask_u8(stream, out, rows, cols, diagonal)
}

pub fn causal_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), len)?;
    let stream = borrowed_stream(stream)?;
    cutile::mask::causal_mask_u8(&stream, output_pointer(out), rows, cols)
}

macro_rules! causal_mask_fill_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            batch: usize,
            seq_len: usize,
            fill_value: $ty,
        ) -> Result<()> {
            validate_causal_score_mask(out.len(), input.len(), batch, seq_len)?;
            let stream = borrowed_stream(stream)?;
            cutile::mask::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                batch,
                seq_len,
                fill_value,
            )
        }
    };
}

macro_rules! causal_mask_zero_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            batch: usize,
            seq_len: usize,
        ) -> Result<()> {
            validate_causal_score_mask(out.len(), input.len(), batch, seq_len)?;
            let stream = borrowed_stream(stream)?;
            cutile::mask::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                batch,
                seq_len,
            )
        }
    };
}

#[cfg(feature = "dtype-f32")]
causal_mask_fill_fn!(causal_mask_fill_f32, f32);
#[cfg(feature = "dtype-f32")]
causal_mask_zero_fn!(causal_mask_zero_f32, f32);
#[cfg(feature = "dtype-f16")]
causal_mask_fill_fn!(causal_mask_fill_f16, f16);
#[cfg(feature = "dtype-f16")]
causal_mask_zero_fn!(causal_mask_zero_f16, f16);
#[cfg(feature = "dtype-bf16")]
causal_mask_fill_fn!(causal_mask_fill_bf16, bf16);
#[cfg(feature = "dtype-bf16")]
causal_mask_zero_fn!(causal_mask_zero_bf16, bf16);

pub fn sequence_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    lengths: &impl DeviceSlice<u32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), len)?;
    ensure_len(lengths.len(), rows)?;
    let stream = borrowed_stream(stream)?;
    cutile::mask::sequence_mask_u8(
        &stream,
        output_pointer(out),
        input_pointer(lengths),
        rows,
        cols,
    )
}

pub fn tril_sequence_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    lengths: &impl DeviceSlice<u32>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    triangular_sequence_mask(
        stream,
        out,
        lengths,
        rows,
        cols,
        diagonal,
        cutile::mask::tril_sequence_mask_u8,
    )
}

pub fn triu_sequence_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    lengths: &impl DeviceSlice<u32>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    triangular_sequence_mask(
        stream,
        out,
        lengths,
        rows,
        cols,
        diagonal,
        cutile::mask::triu_sequence_mask_u8,
    )
}

pub fn lower_triangular_sequence_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    lengths: &impl DeviceSlice<u32>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    tril_sequence_mask_u8(stream, out, lengths, rows, cols, diagonal)
}

pub fn upper_triangular_sequence_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    lengths: &impl DeviceSlice<u32>,
    rows: usize,
    cols: usize,
    diagonal: isize,
) -> Result<()> {
    triu_sequence_mask_u8(stream, out, lengths, rows, cols, diagonal)
}

pub fn causal_sequence_mask_u8(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    lengths: &impl DeviceSlice<u32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), len)?;
    ensure_len(lengths.len(), rows)?;
    let stream = borrowed_stream(stream)?;
    cutile::mask::causal_sequence_mask_u8(
        &stream,
        output_pointer(out),
        input_pointer(lengths),
        rows,
        cols,
    )
}

fn validate_causal_score_mask(
    out_len: usize,
    input_len: usize,
    batch: usize,
    seq_len: usize,
) -> Result<()> {
    if seq_len == 0 {
        return Err(Error::InvalidLength);
    }
    let matrix_len = checked_element_count(seq_len, seq_len)?;
    let len = checked_element_count(batch, matrix_len)?;
    ensure_len(out_len, len)?;
    ensure_len(input_len, len)
}

fn mask_binary<F>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    lhs: &impl DeviceSlice<u8>,
    rhs: &impl DeviceSlice<u8>,
    launch: F,
) -> Result<()>
where
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<u8>,
        DevicePointer<u8>,
        DevicePointer<u8>,
        usize,
    ) -> Result<()>,
{
    let len = out.len();
    ensure_len(lhs.len(), len)?;
    ensure_len(rhs.len(), len)?;
    let stream = borrowed_stream(stream)?;
    launch(
        &stream,
        output_pointer(out),
        input_pointer(lhs),
        input_pointer(rhs),
        len,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_score_mask_validation_accepts_exact_lengths() -> Result<()> {
        validate_causal_score_mask(72, 72, 2, 6)
    }

    #[test]
    fn causal_score_mask_validation_rejects_short_output() {
        assert!(matches!(
            validate_causal_score_mask(71, 72, 2, 6),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn causal_score_mask_validation_rejects_short_input() {
        assert!(matches!(
            validate_causal_score_mask(72, 71, 2, 6),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn causal_score_mask_validation_rejects_zero_sequence() {
        assert!(matches!(
            validate_causal_score_mask(0, 0, 1, 0),
            Err(Error::InvalidLength)
        ));
    }
}

fn mask_reduction<F>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    input: &impl DeviceSlice<u8>,
    rows: usize,
    cols: usize,
    launch: F,
) -> Result<()>
where
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<u8>,
        DevicePointer<u8>,
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
    let stream = borrowed_stream(stream)?;
    launch(
        &stream,
        output_pointer(out),
        input_pointer(input),
        rows,
        cols,
    )
}

fn triangular_mask<F>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    rows: usize,
    cols: usize,
    diagonal: isize,
    launch: F,
) -> Result<()>
where
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<u8>,
        usize,
        usize,
        isize,
    ) -> Result<()>,
{
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), len)?;
    let stream = borrowed_stream(stream)?;
    launch(&stream, output_pointer(out), rows, cols, diagonal)
}

fn triangular_sequence_mask<F>(
    stream: &Stream,
    out: &mut impl DeviceSliceMut<u8>,
    lengths: &impl DeviceSlice<u32>,
    rows: usize,
    cols: usize,
    diagonal: isize,
    launch: F,
) -> Result<()>
where
    F: FnOnce(
        &Arc<::cutile::cuda_core::Stream>,
        DevicePointer<u8>,
        DevicePointer<u32>,
        usize,
        usize,
        isize,
    ) -> Result<()>,
{
    let len = checked_element_count(rows, cols)?;
    ensure_len(out.len(), len)?;
    ensure_len(lengths.len(), rows)?;
    let stream = borrowed_stream(stream)?;
    launch(
        &stream,
        output_pointer(out),
        input_pointer(lengths),
        rows,
        cols,
        diagonal,
    )
}
