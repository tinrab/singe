//! Sum, max, norm, and row/axis reductions.

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

macro_rules! reduction_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            input: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            let out_len = checked_element_count(rows, 2)?;
            ensure_len(out.len(), out_len)?;
            ensure_len(input.len(), len)?;
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let stream = borrowed_stream(stream)?;
            cutile::reduction::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                rows,
                cols,
            )
        }
    };
}

#[cfg(feature = "dtype-f32")]
reduction_fn!(sum_f32, f32);
#[cfg(feature = "dtype-f32")]
reduction_fn!(mean_f32, f32);
#[cfg(feature = "dtype-f32")]
reduction_fn!(variance_f32, f32);
#[cfg(feature = "dtype-f32")]
reduction_fn!(std_f32, f32);
#[cfg(feature = "dtype-f32")]
reduction_fn!(max_f32, f32);
#[cfg(feature = "dtype-f32")]
reduction_fn!(min_f32, f32);
#[cfg(feature = "dtype-f32")]
reduction_fn!(product_f32, f32);

#[cfg(feature = "dtype-f16")]
reduction_fn!(sum_f16, f16);
#[cfg(feature = "dtype-f16")]
reduction_fn!(mean_f16, f16);
#[cfg(feature = "dtype-f16")]
reduction_fn!(variance_f16, f16);
#[cfg(feature = "dtype-f16")]
reduction_fn!(std_f16, f16);
#[cfg(feature = "dtype-f16")]
reduction_fn!(max_f16, f16);
#[cfg(feature = "dtype-f16")]
reduction_fn!(min_f16, f16);
#[cfg(feature = "dtype-f16")]
reduction_fn!(product_f16, f16);

#[cfg(feature = "dtype-f64")]
reduction_fn!(sum_f64, f64);
#[cfg(feature = "dtype-f64")]
reduction_fn!(mean_f64, f64);
#[cfg(feature = "dtype-f64")]
reduction_fn!(variance_f64, f64);
#[cfg(feature = "dtype-f64")]
reduction_fn!(std_f64, f64);
#[cfg(feature = "dtype-f64")]
reduction_fn!(max_f64, f64);
#[cfg(feature = "dtype-f64")]
reduction_fn!(min_f64, f64);
#[cfg(feature = "dtype-f64")]
reduction_fn!(product_f64, f64);

#[cfg(feature = "dtype-u8")]
reduction_fn!(sum_u8, u8);
#[cfg(feature = "dtype-u8")]
reduction_fn!(max_u8, u8);
#[cfg(feature = "dtype-u8")]
reduction_fn!(min_u8, u8);

#[cfg(feature = "dtype-i8")]
reduction_fn!(sum_i8, i8);
#[cfg(feature = "dtype-i8")]
reduction_fn!(max_i8, i8);
#[cfg(feature = "dtype-i8")]
reduction_fn!(min_i8, i8);

#[cfg(feature = "dtype-u32")]
reduction_fn!(sum_u32, u32);
#[cfg(feature = "dtype-u32")]
reduction_fn!(max_u32, u32);
#[cfg(feature = "dtype-u32")]
reduction_fn!(min_u32, u32);

#[cfg(feature = "dtype-i32")]
reduction_fn!(sum_i32, i32);
#[cfg(feature = "dtype-i32")]
reduction_fn!(max_i32, i32);
#[cfg(feature = "dtype-i32")]
reduction_fn!(min_i32, i32);

#[cfg(feature = "dtype-u64")]
reduction_fn!(sum_u64, u64);
#[cfg(feature = "dtype-u64")]
reduction_fn!(max_u64, u64);
#[cfg(feature = "dtype-u64")]
reduction_fn!(min_u64, u64);

#[cfg(feature = "dtype-i64")]
reduction_fn!(sum_i64, i64);
#[cfg(feature = "dtype-i64")]
reduction_fn!(max_i64, i64);
#[cfg(feature = "dtype-i64")]
reduction_fn!(min_i64, i64);

macro_rules! arg_reduction_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<i32>,
            input: &impl DeviceSlice<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            let out_len = checked_element_count(rows, 2)?;
            ensure_len(out.len(), out_len)?;
            ensure_len(input.len(), len)?;
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let stream = borrowed_stream(stream)?;
            cutile::reduction::$name(
                &stream,
                output_pointer(out),
                input_pointer(input),
                rows,
                cols,
            )
        }
    };
}

#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
arg_reduction_fn!(argmax_f32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
arg_reduction_fn!(argmin_f32, f32);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i32"))]
arg_reduction_fn!(argmax_f16, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i32"))]
arg_reduction_fn!(argmin_f16, f16);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
arg_reduction_fn!(argmax_f64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
arg_reduction_fn!(argmin_f64, f64);

#[cfg(all(feature = "dtype-u8", feature = "dtype-i32"))]
arg_reduction_fn!(argmax_u8, u8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-i32"))]
arg_reduction_fn!(argmin_u8, u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-i32"))]
arg_reduction_fn!(argmax_i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-i32"))]
arg_reduction_fn!(argmin_i8, i8);
#[cfg(all(feature = "dtype-u32", feature = "dtype-i32"))]
arg_reduction_fn!(argmax_u32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-i32"))]
arg_reduction_fn!(argmin_u32, u32);
#[cfg(feature = "dtype-i32")]
arg_reduction_fn!(argmax_i32, i32);
#[cfg(feature = "dtype-i32")]
arg_reduction_fn!(argmin_i32, i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-i32"))]
arg_reduction_fn!(argmax_u64, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-i32"))]
arg_reduction_fn!(argmin_u64, u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-i32"))]
arg_reduction_fn!(argmax_i64, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-i32"))]
arg_reduction_fn!(argmin_i64, i64);

macro_rules! block_argmax_fn {
    ($name:ident, $ty:ty, $max_ty:ty) => {
        pub fn $name(
            stream: &Stream,
            block_max: &mut impl DeviceSliceMut<$max_ty>,
            block_idx: &mut impl DeviceSliceMut<i32>,
            input: &impl DeviceSlice<$ty>,
        ) -> Result<usize> {
            let len = input.len();
            if len == 0 {
                return Err(Error::InvalidLength);
            }
            let num_blocks = len.div_ceil(256);
            ensure_len(block_max.len(), num_blocks)?;
            ensure_len(block_idx.len(), num_blocks)?;
            let stream = borrowed_stream(stream)?;
            cutile::reduction::$name(
                &stream,
                output_pointer(block_max),
                output_pointer(block_idx),
                input_pointer(input),
                len,
            )
        }
    };
}

#[cfg(all(feature = "dtype-f16", feature = "dtype-f32", feature = "dtype-i32"))]
block_argmax_fn!(block_argmax_f16, f16, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
block_argmax_fn!(block_argmax_f32, f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
block_argmax_fn!(block_argmax_f64, f64, f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-f32", feature = "dtype-i32"))]
block_argmax_fn!(block_argmin_f16, f16, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
block_argmax_fn!(block_argmin_f32, f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
block_argmax_fn!(block_argmin_f64, f64, f64);

macro_rules! reduce_block_argmax_fn {
    ($name:ident, $max_ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<i32>,
            block_max: &impl DeviceSlice<$max_ty>,
            block_idx: &impl DeviceSlice<i32>,
            num_blocks: usize,
        ) -> Result<()> {
            if num_blocks == 0 {
                return Err(Error::InvalidLength);
            }
            ensure_len(out.len(), 1)?;
            ensure_len(block_max.len(), num_blocks)?;
            ensure_len(block_idx.len(), num_blocks)?;
            let stream = borrowed_stream(stream)?;
            cutile::reduction::$name(
                &stream,
                output_pointer(out),
                input_pointer(block_max),
                input_pointer(block_idx),
                num_blocks,
            )
        }
    };
}

#[cfg(all(feature = "dtype-f16", feature = "dtype-f32", feature = "dtype-i32"))]
reduce_block_argmax_fn!(reduce_block_argmax_f16, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
reduce_block_argmax_fn!(reduce_block_argmax_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
reduce_block_argmax_fn!(reduce_block_argmax_f64, f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-f32", feature = "dtype-i32"))]
reduce_block_argmax_fn!(reduce_block_argmin_f16, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
reduce_block_argmax_fn!(reduce_block_argmin_f32, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
reduce_block_argmax_fn!(reduce_block_argmin_f64, f64);
