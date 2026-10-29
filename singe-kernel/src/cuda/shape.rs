//! Copies, transposes, padding, slicing, tiling, and layout transforms.

use ::cutile::cuda_async::device_buffer::DevicePointer;
#[cfg(feature = "dtype_f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::input_pointer,
    error::{Error, Result},
    utility::{
        checked_element_count, checked_rank4_len, ensure_len, ensure_rank2_reach,
        ensure_rank3_reach, ensure_rank4_reach,
    },
};

macro_rules! copy_fn {
    ($name:ident, $ty:ty) => {
        /// Copies every input element into an equally sized contiguous output.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
        ) -> Result<()> {
            ensure_len(input.len(), out.len())?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    out.len(),
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
copy_fn!(copy_f32, f32);

#[cfg(feature = "dtype_f16")]
copy_fn!(copy_f16, f16);

#[cfg(feature = "dtype_f64")]
copy_fn!(copy_f64, f64);

#[cfg(feature = "dtype_u8")]
copy_fn!(copy_u8, u8);

#[cfg(feature = "dtype_i8")]
copy_fn!(copy_i8, i8);

#[cfg(feature = "dtype_u32")]
copy_fn!(copy_u32, u32);

#[cfg(feature = "dtype_u64")]
copy_fn!(copy_u64, u64);

#[cfg(feature = "dtype_i32")]
copy_fn!(copy_i32, i32);

#[cfg(feature = "dtype_i64")]
copy_fn!(copy_i64, i64);

macro_rules! transpose_2d_fn {
    ($name:ident, $ty:ty) => {
        /// Transposes a contiguous row-major matrix into contiguous output.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, cols)?;
            ensure_len(out.len(), len)?;
            ensure_len(input.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    rows,
                    cols,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
transpose_2d_fn!(transpose_2d_f32, f32);

#[cfg(feature = "dtype_f16")]
transpose_2d_fn!(transpose_2d_f16, f16);

#[cfg(feature = "dtype_f64")]
transpose_2d_fn!(transpose_2d_f64, f64);

#[cfg(feature = "dtype_u8")]
transpose_2d_fn!(transpose_2d_u8, u8);

#[cfg(feature = "dtype_i8")]
transpose_2d_fn!(transpose_2d_i8, i8);

#[cfg(feature = "dtype_u32")]
transpose_2d_fn!(transpose_2d_u32, u32);

#[cfg(feature = "dtype_i32")]
transpose_2d_fn!(transpose_2d_i32, i32);

#[cfg(feature = "dtype_u64")]
transpose_2d_fn!(transpose_2d_u64, u64);

#[cfg(feature = "dtype_i64")]
transpose_2d_fn!(transpose_2d_i64, i64);

macro_rules! transpose_last2_rank3_fn {
    ($name:ident, $ty:ty) => {
        /// Swaps the final two axes of a contiguous rank-3 tensor.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            batch: usize,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = checked_element_count(batch, rows)?;
            let len = checked_element_count(len, cols)?;
            ensure_len(out.len(), len)?;
            ensure_len(input.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    batch,
                    rows,
                    cols,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
transpose_last2_rank3_fn!(transpose_last2_rank3_f32, f32);

#[cfg(feature = "dtype_f16")]
transpose_last2_rank3_fn!(transpose_last2_rank3_f16, f16);

#[cfg(feature = "dtype_f64")]
transpose_last2_rank3_fn!(transpose_last2_rank3_f64, f64);

#[cfg(feature = "dtype_u8")]
transpose_last2_rank3_fn!(transpose_last2_rank3_u8, u8);

#[cfg(feature = "dtype_i8")]
transpose_last2_rank3_fn!(transpose_last2_rank3_i8, i8);

#[cfg(feature = "dtype_u32")]
transpose_last2_rank3_fn!(transpose_last2_rank3_u32, u32);

#[cfg(feature = "dtype_i32")]
transpose_last2_rank3_fn!(transpose_last2_rank3_i32, i32);

#[cfg(feature = "dtype_u64")]
transpose_last2_rank3_fn!(transpose_last2_rank3_u64, u64);

#[cfg(feature = "dtype_i64")]
transpose_last2_rank3_fn!(transpose_last2_rank3_i64, i64);

macro_rules! transpose_last2_rank4_fn {
    ($name:ident, $ty:ty) => {
        /// Swaps the final two axes of a contiguous rank-4 tensor.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            dimensions: [usize; 4],
        ) -> Result<()> {
            let len = checked_rank4_len(dimensions)?;
            ensure_len(out.len(), len)?;
            ensure_len(input.len(), len)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    dimensions,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
transpose_last2_rank4_fn!(transpose_last2_rank4_f32, f32);

#[cfg(feature = "dtype_f16")]
transpose_last2_rank4_fn!(transpose_last2_rank4_f16, f16);

#[cfg(feature = "dtype_f64")]
transpose_last2_rank4_fn!(transpose_last2_rank4_f64, f64);

#[cfg(feature = "dtype_u8")]
transpose_last2_rank4_fn!(transpose_last2_rank4_u8, u8);

#[cfg(feature = "dtype_i8")]
transpose_last2_rank4_fn!(transpose_last2_rank4_i8, i8);

#[cfg(feature = "dtype_u32")]
transpose_last2_rank4_fn!(transpose_last2_rank4_u32, u32);

#[cfg(feature = "dtype_i32")]
transpose_last2_rank4_fn!(transpose_last2_rank4_i32, i32);

#[cfg(feature = "dtype_u64")]
transpose_last2_rank4_fn!(transpose_last2_rank4_u64, u64);

#[cfg(feature = "dtype_i64")]
transpose_last2_rank4_fn!(transpose_last2_rank4_i64, i64);

macro_rules! materialize_rank2_fn {
    ($name:ident, $ty:ty) => {
        /// Copies a strided rank-2 input into contiguous row-major output.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            dimensions: [usize; 2],
            input_strides: [usize; 2],
        ) -> Result<()> {
            let len = checked_element_count(dimensions[0], dimensions[1])?;
            ensure_len(out.len(), len)?;
            ensure_rank2_reach(input.len(), dimensions, input_strides)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    dimensions,
                    input_strides,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
materialize_rank2_fn!(materialize_rank2_f32, f32);

#[cfg(feature = "dtype_f16")]
materialize_rank2_fn!(materialize_rank2_f16, f16);

#[cfg(feature = "dtype_f64")]
materialize_rank2_fn!(materialize_rank2_f64, f64);

#[cfg(feature = "dtype_u8")]
materialize_rank2_fn!(materialize_rank2_u8, u8);

#[cfg(feature = "dtype_i8")]
materialize_rank2_fn!(materialize_rank2_i8, i8);

#[cfg(feature = "dtype_u32")]
materialize_rank2_fn!(materialize_rank2_u32, u32);

#[cfg(feature = "dtype_i32")]
materialize_rank2_fn!(materialize_rank2_i32, i32);

#[cfg(feature = "dtype_u64")]
materialize_rank2_fn!(materialize_rank2_u64, u64);

#[cfg(feature = "dtype_i64")]
materialize_rank2_fn!(materialize_rank2_i64, i64);

macro_rules! slice_rank2_fn {
    ($name:ident, $ty:ty) => {
        /// Copies a rank-2 region at the given offsets into contiguous output.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            output_dimensions: [usize; 2],
            input_dimensions: [usize; 2],
            input_strides: [usize; 2],
            starts: [usize; 2],
        ) -> Result<()> {
            let output_len = checked_element_count(output_dimensions[0], output_dimensions[1])?;
            ensure_len(out.len(), output_len)?;
            ensure_rank2_reach(input.len(), input_dimensions, input_strides)?;
            ensure_slice_bounds(output_dimensions, input_dimensions, starts)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    output_dimensions,
                    input_dimensions,
                    input_strides,
                    starts,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
slice_rank2_fn!(slice_rank2_f32, f32);

#[cfg(feature = "dtype_f16")]
slice_rank2_fn!(slice_rank2_f16, f16);

#[cfg(feature = "dtype_f64")]
slice_rank2_fn!(slice_rank2_f64, f64);

#[cfg(feature = "dtype_u8")]
slice_rank2_fn!(slice_rank2_u8, u8);

#[cfg(feature = "dtype_i8")]
slice_rank2_fn!(slice_rank2_i8, i8);

#[cfg(feature = "dtype_u32")]
slice_rank2_fn!(slice_rank2_u32, u32);

#[cfg(feature = "dtype_i32")]
slice_rank2_fn!(slice_rank2_i32, i32);

#[cfg(feature = "dtype_u64")]
slice_rank2_fn!(slice_rank2_u64, u64);

#[cfg(feature = "dtype_i64")]
slice_rank2_fn!(slice_rank2_i64, i64);

macro_rules! materialize_rank3_fn {
    ($name:ident, $ty:ty) => {
        /// Copies a strided rank-3 input into contiguous row-major output.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            dimensions: [usize; 3],
            input_strides: [usize; 3],
        ) -> Result<()> {
            let len = checked_element_count(dimensions[0], dimensions[1])?;
            let len = checked_element_count(len, dimensions[2])?;
            ensure_len(out.len(), len)?;
            ensure_rank3_reach(input.len(), dimensions, input_strides)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    dimensions,
                    input_strides,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
materialize_rank3_fn!(materialize_rank3_f32, f32);

#[cfg(feature = "dtype_f16")]
materialize_rank3_fn!(materialize_rank3_f16, f16);

#[cfg(feature = "dtype_f64")]
materialize_rank3_fn!(materialize_rank3_f64, f64);

#[cfg(feature = "dtype_u8")]
materialize_rank3_fn!(materialize_rank3_u8, u8);

#[cfg(feature = "dtype_i8")]
materialize_rank3_fn!(materialize_rank3_i8, i8);

#[cfg(feature = "dtype_u32")]
materialize_rank3_fn!(materialize_rank3_u32, u32);

#[cfg(feature = "dtype_i32")]
materialize_rank3_fn!(materialize_rank3_i32, i32);

#[cfg(feature = "dtype_u64")]
materialize_rank3_fn!(materialize_rank3_u64, u64);

#[cfg(feature = "dtype_i64")]
materialize_rank3_fn!(materialize_rank3_i64, i64);

macro_rules! slice_rank3_fn {
    ($name:ident, $ty:ty) => {
        /// Copies a rank-3 region at the given offsets into contiguous output.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            output_dimensions: [usize; 3],
            input_dimensions: [usize; 3],
            input_strides: [usize; 3],
            starts: [usize; 3],
        ) -> Result<()> {
            let output_len = checked_element_count(output_dimensions[0], output_dimensions[1])?;
            let output_len = checked_element_count(output_len, output_dimensions[2])?;
            ensure_len(out.len(), output_len)?;
            ensure_rank3_reach(input.len(), input_dimensions, input_strides)?;
            ensure_slice_bounds(output_dimensions, input_dimensions, starts)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    output_dimensions,
                    input_dimensions,
                    input_strides,
                    starts,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
slice_rank3_fn!(slice_rank3_f32, f32);

#[cfg(feature = "dtype_f16")]
slice_rank3_fn!(slice_rank3_f16, f16);

#[cfg(feature = "dtype_f64")]
slice_rank3_fn!(slice_rank3_f64, f64);

#[cfg(feature = "dtype_u8")]
slice_rank3_fn!(slice_rank3_u8, u8);

#[cfg(feature = "dtype_i8")]
slice_rank3_fn!(slice_rank3_i8, i8);

#[cfg(feature = "dtype_u32")]
slice_rank3_fn!(slice_rank3_u32, u32);

#[cfg(feature = "dtype_i32")]
slice_rank3_fn!(slice_rank3_i32, i32);

#[cfg(feature = "dtype_u64")]
slice_rank3_fn!(slice_rank3_u64, u64);

#[cfg(feature = "dtype_i64")]
slice_rank3_fn!(slice_rank3_i64, i64);

macro_rules! pad_rank2_fn {
    ($name:ident, $ty:ty) => {
        /// Copies a strided rank-2 input into padded contiguous output, filling padding with `pad_value`.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            output_dimensions: [usize; 2],
            input_dimensions: [usize; 2],
            input_strides: [usize; 2],
            pad_before: [usize; 2],
            pad_value: $ty,
        ) -> Result<()> {
            let output_len = checked_element_count(output_dimensions[0], output_dimensions[1])?;
            ensure_len(out.len(), output_len)?;
            ensure_rank2_reach(input.len(), input_dimensions, input_strides)?;
            ensure_pad_bounds(output_dimensions, input_dimensions, pad_before)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    output_dimensions,
                    input_dimensions,
                    input_strides,
                    pad_before,
                    pad_value,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
pad_rank2_fn!(pad_rank2_f32, f32);

#[cfg(feature = "dtype_f16")]
pad_rank2_fn!(pad_rank2_f16, f16);

#[cfg(feature = "dtype_f64")]
pad_rank2_fn!(pad_rank2_f64, f64);

#[cfg(feature = "dtype_u8")]
pad_rank2_fn!(pad_rank2_u8, u8);

#[cfg(feature = "dtype_i8")]
pad_rank2_fn!(pad_rank2_i8, i8);

#[cfg(feature = "dtype_u32")]
pad_rank2_fn!(pad_rank2_u32, u32);

#[cfg(feature = "dtype_i32")]
pad_rank2_fn!(pad_rank2_i32, i32);

#[cfg(feature = "dtype_u64")]
pad_rank2_fn!(pad_rank2_u64, u64);

#[cfg(feature = "dtype_i64")]
pad_rank2_fn!(pad_rank2_i64, i64);

macro_rules! pad_rank3_fn {
    ($name:ident, $ty:ty) => {
        /// Copies a strided rank-3 input into padded contiguous output, filling padding with `pad_value`.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            output_dimensions: [usize; 3],
            input_dimensions: [usize; 3],
            input_strides: [usize; 3],
            pad_before: [usize; 3],
            pad_value: $ty,
        ) -> Result<()> {
            let output_len = checked_element_count(output_dimensions[0], output_dimensions[1])?;
            let output_len = checked_element_count(output_len, output_dimensions[2])?;
            ensure_len(out.len(), output_len)?;
            ensure_rank3_reach(input.len(), input_dimensions, input_strides)?;
            ensure_pad_bounds(output_dimensions, input_dimensions, pad_before)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    output_dimensions,
                    input_dimensions,
                    input_strides,
                    pad_before,
                    pad_value,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
pad_rank3_fn!(pad_rank3_f32, f32);

#[cfg(feature = "dtype_f16")]
pad_rank3_fn!(pad_rank3_f16, f16);

#[cfg(feature = "dtype_f64")]
pad_rank3_fn!(pad_rank3_f64, f64);

#[cfg(feature = "dtype_u8")]
pad_rank3_fn!(pad_rank3_u8, u8);

#[cfg(feature = "dtype_i8")]
pad_rank3_fn!(pad_rank3_i8, i8);

#[cfg(feature = "dtype_u32")]
pad_rank3_fn!(pad_rank3_u32, u32);

#[cfg(feature = "dtype_i32")]
pad_rank3_fn!(pad_rank3_i32, i32);

#[cfg(feature = "dtype_u64")]
pad_rank3_fn!(pad_rank3_u64, u64);

#[cfg(feature = "dtype_i64")]
pad_rank3_fn!(pad_rank3_i64, i64);

macro_rules! materialize_rank4_fn {
    ($name:ident, $ty:ty) => {
        /// Copies a strided rank-4 input into contiguous row-major output.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            dimensions: [usize; 4],
            input_strides: [usize; 4],
        ) -> Result<()> {
            let len = checked_rank4_len(dimensions)?;
            ensure_len(out.len(), len)?;
            ensure_rank4_reach(input.len(), dimensions, input_strides)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    dimensions,
                    input_strides,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
materialize_rank4_fn!(materialize_rank4_f32, f32);

#[cfg(feature = "dtype_f16")]
materialize_rank4_fn!(materialize_rank4_f16, f16);

#[cfg(feature = "dtype_f64")]
materialize_rank4_fn!(materialize_rank4_f64, f64);

#[cfg(feature = "dtype_u8")]
materialize_rank4_fn!(materialize_rank4_u8, u8);

#[cfg(feature = "dtype_i8")]
materialize_rank4_fn!(materialize_rank4_i8, i8);

#[cfg(feature = "dtype_u32")]
materialize_rank4_fn!(materialize_rank4_u32, u32);

#[cfg(feature = "dtype_i32")]
materialize_rank4_fn!(materialize_rank4_i32, i32);

#[cfg(feature = "dtype_u64")]
materialize_rank4_fn!(materialize_rank4_u64, u64);

#[cfg(feature = "dtype_i64")]
materialize_rank4_fn!(materialize_rank4_i64, i64);

macro_rules! pad_rank4_fn {
    ($name:ident, $ty:ty) => {
        /// Copies a strided rank-4 input into padded contiguous output, filling padding with `pad_value`.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            output_dimensions: [usize; 4],
            input_dimensions: [usize; 4],
            input_strides: [usize; 4],
            pad_before: [usize; 4],
            pad_value: $ty,
        ) -> Result<()> {
            let output_len = checked_rank4_len(output_dimensions)?;
            ensure_len(out.len(), output_len)?;
            ensure_rank4_reach(input.len(), input_dimensions, input_strides)?;
            ensure_pad_bounds(output_dimensions, input_dimensions, pad_before)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    output_dimensions,
                    input_dimensions,
                    input_strides,
                    pad_before,
                    pad_value,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
pad_rank4_fn!(pad_rank4_f32, f32);

#[cfg(feature = "dtype_f16")]
pad_rank4_fn!(pad_rank4_f16, f16);

#[cfg(feature = "dtype_f64")]
pad_rank4_fn!(pad_rank4_f64, f64);

#[cfg(feature = "dtype_u8")]
pad_rank4_fn!(pad_rank4_u8, u8);

#[cfg(feature = "dtype_i8")]
pad_rank4_fn!(pad_rank4_i8, i8);

#[cfg(feature = "dtype_u32")]
pad_rank4_fn!(pad_rank4_u32, u32);

#[cfg(feature = "dtype_i32")]
pad_rank4_fn!(pad_rank4_i32, i32);

#[cfg(feature = "dtype_u64")]
pad_rank4_fn!(pad_rank4_u64, u64);

#[cfg(feature = "dtype_i64")]
pad_rank4_fn!(pad_rank4_i64, i64);

macro_rules! slice_rank4_fn {
    ($name:ident, $ty:ty) => {
        /// Copies a rank-4 region at the given offsets into contiguous output.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            output_dimensions: [usize; 4],
            input_dimensions: [usize; 4],
            input_strides: [usize; 4],
            starts: [usize; 4],
        ) -> Result<()> {
            let output_len = checked_rank4_len(output_dimensions)?;
            ensure_len(out.len(), output_len)?;
            ensure_rank4_reach(input.len(), input_dimensions, input_strides)?;
            ensure_slice_bounds(output_dimensions, input_dimensions, starts)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    output_dimensions,
                    input_dimensions,
                    input_strides,
                    starts,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
slice_rank4_fn!(slice_rank4_f32, f32);

#[cfg(feature = "dtype_f16")]
slice_rank4_fn!(slice_rank4_f16, f16);

#[cfg(feature = "dtype_f64")]
slice_rank4_fn!(slice_rank4_f64, f64);

#[cfg(feature = "dtype_u8")]
slice_rank4_fn!(slice_rank4_u8, u8);

#[cfg(feature = "dtype_i8")]
slice_rank4_fn!(slice_rank4_i8, i8);

#[cfg(feature = "dtype_u32")]
slice_rank4_fn!(slice_rank4_u32, u32);

#[cfg(feature = "dtype_i32")]
slice_rank4_fn!(slice_rank4_i32, i32);

#[cfg(feature = "dtype_u64")]
slice_rank4_fn!(slice_rank4_u64, u64);

#[cfg(feature = "dtype_i64")]
slice_rank4_fn!(slice_rank4_i64, i64);

macro_rules! concat_rank2_fn {
    ($name:ident, $ty:ty) => {
        /// Concatenates two strided rank-2 inputs along `axis` into contiguous output.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            output_dimensions: [usize; 2],
            lhs_dimensions: [usize; 2],
            lhs_strides: [usize; 2],
            rhs_strides: [usize; 2],
            axis: usize,
        ) -> Result<()> {
            ensure_concat_layout(output_dimensions, lhs_dimensions, axis)?;
            let rhs_dimensions = rhs_dimensions(output_dimensions, lhs_dimensions, axis)?;
            let output_len = checked_element_count(output_dimensions[0], output_dimensions[1])?;
            ensure_len(out.len(), output_len)?;
            ensure_rank2_reach(lhs.len(), lhs_dimensions, lhs_strides)?;
            ensure_rank2_reach(rhs.len(), rhs_dimensions, rhs_strides)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(lhs),
                    input_pointer(rhs),
                    output_dimensions,
                    lhs_dimensions,
                    lhs_strides,
                    rhs_strides,
                    axis,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
concat_rank2_fn!(concat_rank2_f32, f32);

#[cfg(feature = "dtype_f16")]
concat_rank2_fn!(concat_rank2_f16, f16);

#[cfg(feature = "dtype_f64")]
concat_rank2_fn!(concat_rank2_f64, f64);

#[cfg(feature = "dtype_u8")]
concat_rank2_fn!(concat_rank2_u8, u8);

#[cfg(feature = "dtype_i8")]
concat_rank2_fn!(concat_rank2_i8, i8);

#[cfg(feature = "dtype_u32")]
concat_rank2_fn!(concat_rank2_u32, u32);

#[cfg(feature = "dtype_i32")]
concat_rank2_fn!(concat_rank2_i32, i32);

#[cfg(feature = "dtype_u64")]
concat_rank2_fn!(concat_rank2_u64, u64);

#[cfg(feature = "dtype_i64")]
concat_rank2_fn!(concat_rank2_i64, i64);

macro_rules! concat_rank3_fn {
    ($name:ident, $ty:ty) => {
        /// Concatenates two strided rank-3 inputs along `axis` into contiguous output.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            output_dimensions: [usize; 3],
            lhs_dimensions: [usize; 3],
            lhs_strides: [usize; 3],
            rhs_strides: [usize; 3],
            axis: usize,
        ) -> Result<()> {
            ensure_concat_layout(output_dimensions, lhs_dimensions, axis)?;
            let rhs_dimensions = rhs_dimensions(output_dimensions, lhs_dimensions, axis)?;
            let output_len = checked_element_count(output_dimensions[0], output_dimensions[1])?;
            let output_len = checked_element_count(output_len, output_dimensions[2])?;
            ensure_len(out.len(), output_len)?;
            ensure_rank3_reach(lhs.len(), lhs_dimensions, lhs_strides)?;
            ensure_rank3_reach(rhs.len(), rhs_dimensions, rhs_strides)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(lhs),
                    input_pointer(rhs),
                    output_dimensions,
                    lhs_dimensions,
                    lhs_strides,
                    rhs_strides,
                    axis,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
concat_rank3_fn!(concat_rank3_f32, f32);

#[cfg(feature = "dtype_f16")]
concat_rank3_fn!(concat_rank3_f16, f16);

#[cfg(feature = "dtype_f64")]
concat_rank3_fn!(concat_rank3_f64, f64);

#[cfg(feature = "dtype_u8")]
concat_rank3_fn!(concat_rank3_u8, u8);

#[cfg(feature = "dtype_i8")]
concat_rank3_fn!(concat_rank3_i8, i8);

#[cfg(feature = "dtype_u32")]
concat_rank3_fn!(concat_rank3_u32, u32);

#[cfg(feature = "dtype_i32")]
concat_rank3_fn!(concat_rank3_i32, i32);

#[cfg(feature = "dtype_u64")]
concat_rank3_fn!(concat_rank3_u64, u64);

#[cfg(feature = "dtype_i64")]
concat_rank3_fn!(concat_rank3_i64, i64);

macro_rules! concat_rank4_fn {
    ($name:ident, $ty:ty) => {
        /// Concatenates two strided rank-4 inputs along `axis` into contiguous output.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            lhs: &impl DeviceInput<$ty>,
            rhs: &impl DeviceInput<$ty>,
            output_dimensions: [usize; 4],
            lhs_dimensions: [usize; 4],
            lhs_strides: [usize; 4],
            rhs_strides: [usize; 4],
            axis: usize,
        ) -> Result<()> {
            ensure_concat_layout(output_dimensions, lhs_dimensions, axis)?;
            let rhs_dimensions = rhs_dimensions(output_dimensions, lhs_dimensions, axis)?;
            let output_len = checked_rank4_len(output_dimensions)?;
            ensure_len(out.len(), output_len)?;
            ensure_rank4_reach(lhs.len(), lhs_dimensions, lhs_strides)?;
            ensure_rank4_reach(rhs.len(), rhs_dimensions, rhs_strides)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(lhs),
                    input_pointer(rhs),
                    output_dimensions,
                    lhs_dimensions,
                    lhs_strides,
                    rhs_strides,
                    axis,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
concat_rank4_fn!(concat_rank4_f32, f32);

#[cfg(feature = "dtype_f16")]
concat_rank4_fn!(concat_rank4_f16, f16);

#[cfg(feature = "dtype_f64")]
concat_rank4_fn!(concat_rank4_f64, f64);

#[cfg(feature = "dtype_u8")]
concat_rank4_fn!(concat_rank4_u8, u8);

#[cfg(feature = "dtype_i8")]
concat_rank4_fn!(concat_rank4_i8, i8);

#[cfg(feature = "dtype_u32")]
concat_rank4_fn!(concat_rank4_u32, u32);

#[cfg(feature = "dtype_i32")]
concat_rank4_fn!(concat_rank4_i32, i32);

#[cfg(feature = "dtype_u64")]
concat_rank4_fn!(concat_rank4_u64, u64);

#[cfg(feature = "dtype_i64")]
concat_rank4_fn!(concat_rank4_i64, i64);

macro_rules! repeat_kv_rank4_fn {
    ($name:ident, $ty:ty) => {
        /// Repeats attention KV heads along the head axis into contiguous output.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Every logical output element is written; input stride gaps are not copied.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            input: &impl DeviceInput<$ty>,
            output_dimensions: [usize; 4],
            input_strides: [usize; 4],
            repeats: usize,
        ) -> Result<()> {
            let input_dimensions = repeat_kv_input_dimensions(output_dimensions, repeats)?;
            let output_len = checked_rank4_len(output_dimensions)?;
            ensure_len(out.len(), output_len)?;
            ensure_rank4_reach(input.len(), input_dimensions, input_strides)?;
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::shape::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(input),
                    output_dimensions,
                    input_strides,
                    repeats,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f32")]
repeat_kv_rank4_fn!(repeat_kv_rank4_f32, f32);

#[cfg(feature = "dtype_f16")]
repeat_kv_rank4_fn!(repeat_kv_rank4_f16, f16);

#[cfg(feature = "dtype_f64")]
repeat_kv_rank4_fn!(repeat_kv_rank4_f64, f64);

#[cfg(feature = "dtype_u8")]
repeat_kv_rank4_fn!(repeat_kv_rank4_u8, u8);

#[cfg(feature = "dtype_i8")]
repeat_kv_rank4_fn!(repeat_kv_rank4_i8, i8);

#[cfg(feature = "dtype_u32")]
repeat_kv_rank4_fn!(repeat_kv_rank4_u32, u32);

#[cfg(feature = "dtype_i32")]
repeat_kv_rank4_fn!(repeat_kv_rank4_i32, i32);

#[cfg(feature = "dtype_u64")]
repeat_kv_rank4_fn!(repeat_kv_rank4_u64, u64);

#[cfg(feature = "dtype_i64")]
repeat_kv_rank4_fn!(repeat_kv_rank4_i64, i64);

#[cfg(feature = "dtype_f32")]
/// Packs each group of `factor` input rows into a single wider output row.
///
/// # Safety
///
/// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
/// Input and output ranges must not overlap.
/// Every logical output element is written; input stride gaps are not copied.
pub unsafe fn pack_downsample_2d_f32(
    stream: &Stream,
    out: &mut impl DeviceOutput<f32>,
    input: &impl DeviceInput<f32>,
    rows: usize,
    cols: usize,
    factor: usize,
) -> Result<()> {
    if factor == 0 {
        return Err(Error::InvalidLength);
    }
    if !rows.is_multiple_of(factor) {
        return Err(Error::LengthMismatch);
    }
    let len = checked_element_count(rows, cols)?;
    ensure_len(input.len(), len)?;
    ensure_len(out.len(), len)?;
    let stream = stream.to_cuda_core()?;
    // SAFETY: the caller upholds this function's safety contract.
    unsafe {
        cutile::shape::pack_downsample_2d_f32(
            &stream,
            DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
            input_pointer(input),
            rows,
            cols,
            factor,
        )
    }
}

fn repeat_kv_input_dimensions(output_dimensions: [usize; 4], repeats: usize) -> Result<[usize; 4]> {
    if repeats == 0 {
        return Err(Error::InvalidLength);
    }
    if !output_dimensions[1].is_multiple_of(repeats) {
        return Err(Error::LengthMismatch);
    }
    let mut dimensions = output_dimensions;
    dimensions[1] /= repeats;
    Ok(dimensions)
}

fn ensure_concat_layout<const R: usize>(
    output_dimensions: [usize; R],
    lhs_dimensions: [usize; R],
    axis: usize,
) -> Result<()> {
    if axis >= R {
        return Err(Error::UnsupportedAxis {
            op: "concat".into(),
            axis,
        });
    }
    if lhs_dimensions[axis] > output_dimensions[axis] {
        return Err(Error::LengthMismatch);
    }
    for dim in 0..R {
        if dim != axis && lhs_dimensions[dim] != output_dimensions[dim] {
            return Err(Error::LengthMismatch);
        }
    }
    Ok(())
}

fn rhs_dimensions<const R: usize>(
    output_dimensions: [usize; R],
    lhs_dimensions: [usize; R],
    axis: usize,
) -> Result<[usize; R]> {
    let mut dimensions = output_dimensions;
    dimensions[axis] = output_dimensions[axis]
        .checked_sub(lhs_dimensions[axis])
        .ok_or(Error::LengthMismatch)?;
    Ok(dimensions)
}

fn ensure_pad_bounds<const R: usize>(
    output_dimensions: [usize; R],
    input_dimensions: [usize; R],
    pad_before: [usize; R],
) -> Result<()> {
    for axis in 0..R {
        let input_end = pad_before[axis]
            .checked_add(input_dimensions[axis])
            .ok_or(Error::SizeOverflow)?;
        if input_end > output_dimensions[axis] {
            return Err(Error::LengthMismatch);
        }
    }
    Ok(())
}

fn ensure_slice_bounds<const R: usize>(
    output_dimensions: [usize; R],
    input_dimensions: [usize; R],
    starts: [usize; R],
) -> Result<()> {
    for axis in 0..R {
        let output_end = starts[axis]
            .checked_add(output_dimensions[axis])
            .ok_or(Error::SizeOverflow)?;
        if output_end > input_dimensions[axis] {
            return Err(Error::LengthMismatch);
        }
    }
    Ok(())
}
