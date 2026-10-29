//! Index-based gathers and row selection.

#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::{input_pointer, write_only_pointer},
    error::{Error, Result},
    utility::{checked_element_count, ensure_len},
};

macro_rules! gather_fn {
    ($name:ident, $ty:ty) => {
        /// Writes `out[j] = input[indices[j]]` for every output element.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read, with preceding producers ordered before use.
        /// Retain every allocation in the stream's context through completion on every
        /// path and throughout captured replay. Output ranges must be exclusive,
        /// mutually disjoint, and disjoint from readable input ranges.
        /// Device indices must be in bounds and fit the signed i32 address calculations.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            indices: &impl DeviceInput<u32>,
        ) -> Result<()> {
            let len = out.len();
            ensure_len(indices.len(), len)?;
            if len > 0 && input.len() == 0 {
                return Err(Error::InvalidLength);
            }
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::gather::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(input),
                    input_pointer(indices),
                    len,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f16")]
gather_fn!(gather_f16, f16);

#[cfg(feature = "dtype_f32")]
gather_fn!(gather_f32, f32);

#[cfg(feature = "dtype_f64")]
gather_fn!(gather_f64, f64);

#[cfg(feature = "dtype_u8")]
gather_fn!(gather_u8, u8);

#[cfg(feature = "dtype_i8")]
gather_fn!(gather_i8, i8);

#[cfg(feature = "dtype_u32")]
gather_fn!(gather_u32, u32);

#[cfg(feature = "dtype_i32")]
gather_fn!(gather_i32, i32);

#[cfg(feature = "dtype_u64")]
gather_fn!(gather_u64, u64);

#[cfg(feature = "dtype_i64")]
gather_fn!(gather_i64, i64);

macro_rules! scatter_fn {
    ($name:ident, $ty:ty) => {
        /// Writes `input[j]` to `out[indices[j]]`.
        /// Only selected output elements are produced; all other positions remain untouched.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read, with preceding producers ordered before use.
        /// Retain every allocation in the stream's context through completion on every
        /// path and throughout captured replay. Output ranges must be exclusive,
        /// mutually disjoint, and disjoint from readable input ranges.
        /// Device indices must be in bounds and fit the signed i32 address calculations.
        /// Destination indices must be distinct because the stores are non-atomic.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            indices: &impl DeviceInput<u32>,
        ) -> Result<()> {
            let len = input.len();
            ensure_len(indices.len(), len)?;
            if len > 0 && out.len() == 0 {
                return Err(Error::InvalidLength);
            }
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::gather::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(input),
                    input_pointer(indices),
                    len,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f16")]
scatter_fn!(scatter_f16, f16);

#[cfg(feature = "dtype_f32")]
scatter_fn!(scatter_f32, f32);

#[cfg(feature = "dtype_f64")]
scatter_fn!(scatter_f64, f64);

#[cfg(feature = "dtype_u8")]
scatter_fn!(scatter_u8, u8);

#[cfg(feature = "dtype_i8")]
scatter_fn!(scatter_i8, i8);

#[cfg(feature = "dtype_u32")]
scatter_fn!(scatter_u32, u32);

#[cfg(feature = "dtype_i32")]
scatter_fn!(scatter_i32, i32);

#[cfg(feature = "dtype_u64")]
scatter_fn!(scatter_u64, u64);

#[cfg(feature = "dtype_i64")]
scatter_fn!(scatter_i64, i64);

macro_rules! gather_rows_fn {
    ($name:ident, $ty:ty) => {
        /// Writes complete output rows selected by `row_indices` from the input matrix.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read, with preceding producers ordered before use.
        /// Retain every allocation in the stream's context through completion on every
        /// path and throughout captured replay. Output ranges must be exclusive,
        /// mutually disjoint, and disjoint from readable input ranges.
        /// Device indices must be in bounds and fit the signed i32 address calculations.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            row_indices: &impl DeviceInput<u32>,
            cols: usize,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(row_indices.len(), cols)?;
            ensure_len(out.len(), len)?;
            if !input.len().is_multiple_of(cols) {
                return Err(Error::LengthMismatch);
            }
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::gather::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(input),
                    input_pointer(row_indices),
                    row_indices.len(),
                    cols,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f16")]
gather_rows_fn!(gather_rows_f16, f16);

#[cfg(feature = "dtype_f32")]
gather_rows_fn!(gather_rows_f32, f32);

#[cfg(feature = "dtype_f64")]
gather_rows_fn!(gather_rows_f64, f64);

#[cfg(feature = "dtype_u8")]
gather_rows_fn!(gather_rows_u8, u8);

#[cfg(feature = "dtype_i8")]
gather_rows_fn!(gather_rows_i8, i8);

#[cfg(feature = "dtype_u32")]
gather_rows_fn!(gather_rows_u32, u32);

#[cfg(feature = "dtype_i32")]
gather_rows_fn!(gather_rows_i32, i32);

#[cfg(feature = "dtype_u64")]
gather_rows_fn!(gather_rows_u64, u64);

#[cfg(feature = "dtype_i64")]
gather_rows_fn!(gather_rows_i64, i64);

macro_rules! gather_row_fn {
    ($name:ident, $ty:ty) => {
        /// Copies one complete input row into the output vector.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read, with preceding producers ordered before use.
        /// Retain every allocation in the stream's context through completion on every
        /// path and throughout captured replay. Output ranges must be exclusive,
        /// mutually disjoint, and disjoint from readable input ranges.
        /// Device indices must be in bounds and fit the signed i32 address calculations.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            row_index: usize,
            cols: usize,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            ensure_len(out.len(), cols)?;
            if !input.len().is_multiple_of(cols) {
                return Err(Error::LengthMismatch);
            }
            let rows = input.len() / cols;
            if row_index >= rows {
                return Err(Error::InvalidLength);
            }
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::gather::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(input),
                    row_index,
                    rows,
                    cols,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f16")]
gather_row_fn!(gather_row_f16, f16);

#[cfg(feature = "dtype_f32")]
gather_row_fn!(gather_row_f32, f32);

#[cfg(feature = "dtype_f64")]
gather_row_fn!(gather_row_f64, f64);

#[cfg(feature = "dtype_u8")]
gather_row_fn!(gather_row_u8, u8);

#[cfg(feature = "dtype_i8")]
gather_row_fn!(gather_row_i8, i8);

#[cfg(feature = "dtype_u32")]
gather_row_fn!(gather_row_u32, u32);

#[cfg(feature = "dtype_i32")]
gather_row_fn!(gather_row_i32, i32);

#[cfg(feature = "dtype_u64")]
gather_row_fn!(gather_row_u64, u64);

#[cfg(feature = "dtype_i64")]
gather_row_fn!(gather_row_i64, i64);

macro_rules! scatter_rows_fn {
    ($name:ident, $ty:ty) => {
        /// Writes input rows to the output rows selected by `row_indices`.
        /// Unselected output rows remain untouched and may still be uninitialized.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read, with preceding producers ordered before use.
        /// Retain every allocation in the stream's context through completion on every
        /// path and throughout captured replay. Output ranges must be exclusive,
        /// mutually disjoint, and disjoint from readable input ranges.
        /// Device indices must be in bounds and fit the signed i32 address calculations.
        /// Destination indices must be distinct because the stores are non-atomic.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            row_indices: &impl DeviceInput<u32>,
            cols: usize,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(row_indices.len(), cols)?;
            ensure_len(input.len(), len)?;
            if !out.len().is_multiple_of(cols) {
                return Err(Error::LengthMismatch);
            }
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::gather::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(input),
                    input_pointer(row_indices),
                    row_indices.len(),
                    cols,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f16")]
scatter_rows_fn!(scatter_rows_f16, f16);

#[cfg(feature = "dtype_f32")]
scatter_rows_fn!(scatter_rows_f32, f32);

#[cfg(feature = "dtype_f64")]
scatter_rows_fn!(scatter_rows_f64, f64);

#[cfg(feature = "dtype_u8")]
scatter_rows_fn!(scatter_rows_u8, u8);

#[cfg(feature = "dtype_i8")]
scatter_rows_fn!(scatter_rows_i8, i8);

#[cfg(feature = "dtype_u32")]
scatter_rows_fn!(scatter_rows_u32, u32);

#[cfg(feature = "dtype_i32")]
scatter_rows_fn!(scatter_rows_i32, i32);

#[cfg(feature = "dtype_u64")]
scatter_rows_fn!(scatter_rows_u64, u64);

#[cfg(feature = "dtype_i64")]
scatter_rows_fn!(scatter_rows_i64, i64);

macro_rules! copy_indexed_rows_fn {
    ($name:ident, $ty:ty) => {
        /// Copies each selected source row into its selected output row.
        /// Only the selected output rows are produced.
        ///
        /// # Safety
        ///
        /// Input values must be valid when read, with preceding producers ordered before use.
        /// Retain every allocation in the stream's context through completion on every
        /// path and throughout captured replay. Output ranges must be exclusive,
        /// mutually disjoint, and disjoint from readable input ranges.
        /// Device indices must be in bounds and fit the signed i32 address calculations.
        /// Destination indices must be distinct because the stores are non-atomic.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            source_indices: &impl DeviceInput<u32>,
            output_indices: &impl DeviceInput<u32>,
            cols: usize,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            ensure_len(source_indices.len(), output_indices.len())?;
            let len = checked_element_count(source_indices.len(), cols)?;
            if len == 0 {
                return Ok(());
            }
            if !input.len().is_multiple_of(cols) {
                return Err(Error::LengthMismatch);
            }
            if !out.len().is_multiple_of(cols) {
                return Err(Error::LengthMismatch);
            }
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::gather::$name(
                    &stream,
                    write_only_pointer(out),
                    input_pointer(input),
                    input_pointer(source_indices),
                    input_pointer(output_indices),
                    source_indices.len(),
                    cols,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f16")]
copy_indexed_rows_fn!(copy_indexed_rows_f16, f16);

#[cfg(feature = "dtype_f32")]
copy_indexed_rows_fn!(copy_indexed_rows_f32, f32);

#[cfg(feature = "dtype_f64")]
copy_indexed_rows_fn!(copy_indexed_rows_f64, f64);

#[cfg(feature = "dtype_u8")]
copy_indexed_rows_fn!(copy_indexed_rows_u8, u8);

#[cfg(feature = "dtype_i8")]
copy_indexed_rows_fn!(copy_indexed_rows_i8, i8);

#[cfg(feature = "dtype_u32")]
copy_indexed_rows_fn!(copy_indexed_rows_u32, u32);

#[cfg(feature = "dtype_i32")]
copy_indexed_rows_fn!(copy_indexed_rows_i32, i32);

#[cfg(feature = "dtype_u64")]
copy_indexed_rows_fn!(copy_indexed_rows_u64, u64);

#[cfg(feature = "dtype_i64")]
copy_indexed_rows_fn!(copy_indexed_rows_i64, i64);
