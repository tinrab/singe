//! Index-based gathers and row selection.

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

macro_rules! gather_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            indices: &impl DeviceSlice<u32>,
        ) -> Result<()> {
            let len = out.len();
            ensure_len(indices.len(), len)?;
            if len > 0 && input.len() == 0 {
                return Err(Error::InvalidLength);
            }
            let stream = borrowed_stream(stream)?;
            cutile::gather::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(indices),
                len,
            )
        }
    };
}

#[cfg(feature = "dtype-f16")]
gather_fn!(gather_f16, f16);
#[cfg(feature = "dtype-f32")]
gather_fn!(gather_f32, f32);
#[cfg(feature = "dtype-f64")]
gather_fn!(gather_f64, f64);
#[cfg(feature = "dtype-u8")]
gather_fn!(gather_u8, u8);
#[cfg(feature = "dtype-i8")]
gather_fn!(gather_i8, i8);
#[cfg(feature = "dtype-u32")]
gather_fn!(gather_u32, u32);
#[cfg(feature = "dtype-i32")]
gather_fn!(gather_i32, i32);
#[cfg(feature = "dtype-u64")]
gather_fn!(gather_u64, u64);
#[cfg(feature = "dtype-i64")]
gather_fn!(gather_i64, i64);

macro_rules! scatter_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            indices: &impl DeviceSlice<u32>,
        ) -> Result<()> {
            let len = input.len();
            ensure_len(indices.len(), len)?;
            if len > 0 && out.len() == 0 {
                return Err(Error::InvalidLength);
            }
            let stream = borrowed_stream(stream)?;
            cutile::gather::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(indices),
                len,
            )
        }
    };
}

#[cfg(feature = "dtype-f16")]
scatter_fn!(scatter_f16, f16);
#[cfg(feature = "dtype-f32")]
scatter_fn!(scatter_f32, f32);
#[cfg(feature = "dtype-f64")]
scatter_fn!(scatter_f64, f64);
#[cfg(feature = "dtype-u8")]
scatter_fn!(scatter_u8, u8);
#[cfg(feature = "dtype-i8")]
scatter_fn!(scatter_i8, i8);
#[cfg(feature = "dtype-u32")]
scatter_fn!(scatter_u32, u32);
#[cfg(feature = "dtype-i32")]
scatter_fn!(scatter_i32, i32);
#[cfg(feature = "dtype-u64")]
scatter_fn!(scatter_u64, u64);
#[cfg(feature = "dtype-i64")]
scatter_fn!(scatter_i64, i64);

macro_rules! gather_rows_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            row_indices: &impl DeviceSlice<u32>,
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
            let stream = borrowed_stream(stream)?;
            cutile::gather::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(row_indices),
                row_indices.len(),
                cols,
            )
        }
    };
}

#[cfg(feature = "dtype-f16")]
gather_rows_fn!(gather_rows_f16, f16);
#[cfg(feature = "dtype-f32")]
gather_rows_fn!(gather_rows_f32, f32);
#[cfg(feature = "dtype-f64")]
gather_rows_fn!(gather_rows_f64, f64);
#[cfg(feature = "dtype-u8")]
gather_rows_fn!(gather_rows_u8, u8);
#[cfg(feature = "dtype-i8")]
gather_rows_fn!(gather_rows_i8, i8);
#[cfg(feature = "dtype-u32")]
gather_rows_fn!(gather_rows_u32, u32);
#[cfg(feature = "dtype-i32")]
gather_rows_fn!(gather_rows_i32, i32);
#[cfg(feature = "dtype-u64")]
gather_rows_fn!(gather_rows_u64, u64);
#[cfg(feature = "dtype-i64")]
gather_rows_fn!(gather_rows_i64, i64);

macro_rules! gather_row_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
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
            let stream = borrowed_stream(stream)?;
            cutile::gather::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                row_index,
                rows,
                cols,
            )
        }
    };
}

#[cfg(feature = "dtype-f16")]
gather_row_fn!(gather_row_f16, f16);
#[cfg(feature = "dtype-f32")]
gather_row_fn!(gather_row_f32, f32);
#[cfg(feature = "dtype-f64")]
gather_row_fn!(gather_row_f64, f64);
#[cfg(feature = "dtype-u8")]
gather_row_fn!(gather_row_u8, u8);
#[cfg(feature = "dtype-i8")]
gather_row_fn!(gather_row_i8, i8);
#[cfg(feature = "dtype-u32")]
gather_row_fn!(gather_row_u32, u32);
#[cfg(feature = "dtype-i32")]
gather_row_fn!(gather_row_i32, i32);
#[cfg(feature = "dtype-u64")]
gather_row_fn!(gather_row_u64, u64);
#[cfg(feature = "dtype-i64")]
gather_row_fn!(gather_row_i64, i64);

macro_rules! scatter_rows_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            row_indices: &impl DeviceSlice<u32>,
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
            let stream = borrowed_stream(stream)?;
            cutile::gather::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(row_indices),
                row_indices.len(),
                cols,
            )
        }
    };
}

#[cfg(feature = "dtype-f16")]
scatter_rows_fn!(scatter_rows_f16, f16);
#[cfg(feature = "dtype-f32")]
scatter_rows_fn!(scatter_rows_f32, f32);
#[cfg(feature = "dtype-f64")]
scatter_rows_fn!(scatter_rows_f64, f64);
#[cfg(feature = "dtype-u8")]
scatter_rows_fn!(scatter_rows_u8, u8);
#[cfg(feature = "dtype-i8")]
scatter_rows_fn!(scatter_rows_i8, i8);
#[cfg(feature = "dtype-u32")]
scatter_rows_fn!(scatter_rows_u32, u32);
#[cfg(feature = "dtype-i32")]
scatter_rows_fn!(scatter_rows_i32, i32);
#[cfg(feature = "dtype-u64")]
scatter_rows_fn!(scatter_rows_u64, u64);
#[cfg(feature = "dtype-i64")]
scatter_rows_fn!(scatter_rows_i64, i64);

macro_rules! copy_indexed_rows_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            source_indices: &impl DeviceSlice<u32>,
            output_indices: &impl DeviceSlice<u32>,
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
            let stream = borrowed_stream(stream)?;
            cutile::gather::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                input_pointer(source_indices),
                input_pointer(output_indices),
                source_indices.len(),
                cols,
            )
        }
    };
}

#[cfg(feature = "dtype-f16")]
copy_indexed_rows_fn!(copy_indexed_rows_f16, f16);
#[cfg(feature = "dtype-f32")]
copy_indexed_rows_fn!(copy_indexed_rows_f32, f32);
#[cfg(feature = "dtype-f64")]
copy_indexed_rows_fn!(copy_indexed_rows_f64, f64);
#[cfg(feature = "dtype-u8")]
copy_indexed_rows_fn!(copy_indexed_rows_u8, u8);
#[cfg(feature = "dtype-i8")]
copy_indexed_rows_fn!(copy_indexed_rows_i8, i8);
#[cfg(feature = "dtype-u32")]
copy_indexed_rows_fn!(copy_indexed_rows_u32, u32);
#[cfg(feature = "dtype-i32")]
copy_indexed_rows_fn!(copy_indexed_rows_i32, i32);
#[cfg(feature = "dtype-u64")]
copy_indexed_rows_fn!(copy_indexed_rows_u64, u64);
#[cfg(feature = "dtype-i64")]
copy_indexed_rows_fn!(copy_indexed_rows_i64, i64);
