use std::sync::Arc;

#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

#[cfg(any(
    feature = "dtype-i8",
    feature = "dtype-u8",
    feature = "dtype-i32",
    feature = "dtype-u32",
    feature = "dtype-i64",
    feature = "dtype-u64",
))]
use crate::cuda::cutile::kernel::common as kernel_common;
#[cfg(feature = "dtype-f16")]
use crate::cuda::cutile::kernel::f16::shape as kernel_f16;
#[cfg(feature = "dtype-f32")]
use crate::cuda::cutile::kernel::f32::shape as kernel_f32;
#[cfg(feature = "dtype-f64")]
use crate::cuda::cutile::kernel::f64::shape as kernel_f64;
use crate::{
    cuda::cutile::{
        DeviceOpExt,
        utility::{VectorLaunch, checked_device_pointer},
    },
    error::{Error, Result},
    utility::{
        checked_i32_value, checked_rank2_i32, checked_rank3_i32, checked_rank4_i32,
        checked_rank4_len,
    },
};

macro_rules! copy_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { $kernel::$kernel_fn(out, input, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
copy_fn!(copy_f32, f32, kernel_f32, copy_f32);
#[cfg(feature = "dtype-f16")]
copy_fn!(copy_f16, f16, kernel_f16, copy_f16);
#[cfg(feature = "dtype-f64")]
copy_fn!(copy_f64, f64, kernel_f64, copy_f64);
#[cfg(feature = "dtype-u8")]
copy_fn!(copy_u8, u8, kernel_common, copy_u8);
#[cfg(feature = "dtype-i8")]
copy_fn!(copy_i8, i8, kernel_common, copy_i8);
#[cfg(feature = "dtype-u32")]
copy_fn!(copy_u32, u32, kernel_common, copy_u32);
#[cfg(feature = "dtype-u64")]
copy_fn!(copy_u64, u64, kernel_common, copy_u64);
#[cfg(feature = "dtype-i32")]
copy_fn!(copy_i32, i32, kernel_common, copy_i32);
#[cfg(feature = "dtype-i64")]
copy_fn!(copy_i64, i64, kernel_common, copy_i64);

macro_rules! transpose_2d_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = crate::utility::checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let rows_i32 = checked_i32_value(rows)?;
            let cols_i32 = checked_i32_value(cols)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { $kernel::$kernel_fn(out, input, rows_i32, cols_i32, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
transpose_2d_fn!(transpose_2d_f32, f32, kernel_f32, transpose_2d_f32);
#[cfg(feature = "dtype-f16")]
transpose_2d_fn!(transpose_2d_f16, f16, kernel_f16, transpose_2d_f16);
#[cfg(feature = "dtype-f64")]
transpose_2d_fn!(transpose_2d_f64, f64, kernel_f64, transpose_2d_f64);
#[cfg(feature = "dtype-u8")]
transpose_2d_fn!(transpose_2d_u8, u8, kernel_common, transpose_2d_u8);
#[cfg(feature = "dtype-i8")]
transpose_2d_fn!(transpose_2d_i8, i8, kernel_common, transpose_2d_i8);
#[cfg(feature = "dtype-u32")]
transpose_2d_fn!(transpose_2d_u32, u32, kernel_common, transpose_2d_u32);
#[cfg(feature = "dtype-i32")]
transpose_2d_fn!(transpose_2d_i32, i32, kernel_common, transpose_2d_i32);
#[cfg(feature = "dtype-u64")]
transpose_2d_fn!(transpose_2d_u64, u64, kernel_common, transpose_2d_u64);
#[cfg(feature = "dtype-i64")]
transpose_2d_fn!(transpose_2d_i64, i64, kernel_common, transpose_2d_i64);

macro_rules! transpose_last2_rank3_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            batch: usize,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            let len = crate::utility::checked_element_count(batch, rows)?;
            let len = crate::utility::checked_element_count(len, cols)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let batch_i32 = checked_i32_value(batch)?;
            let rows_i32 = checked_i32_value(rows)?;
            let cols_i32 = checked_i32_value(cols)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(out, input, batch_i32, rows_i32, cols_i32, launch.len_i32)
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
transpose_last2_rank3_fn!(
    transpose_last2_rank3_f32,
    f32,
    kernel_f32,
    transpose_last2_rank3_f32
);
#[cfg(feature = "dtype-f16")]
transpose_last2_rank3_fn!(
    transpose_last2_rank3_f16,
    f16,
    kernel_f16,
    transpose_last2_rank3_f16
);
#[cfg(feature = "dtype-f64")]
transpose_last2_rank3_fn!(
    transpose_last2_rank3_f64,
    f64,
    kernel_f64,
    transpose_last2_rank3_f64
);
#[cfg(feature = "dtype-u8")]
transpose_last2_rank3_fn!(
    transpose_last2_rank3_u8,
    u8,
    kernel_common,
    transpose_last2_rank3_u8
);
#[cfg(feature = "dtype-i8")]
transpose_last2_rank3_fn!(
    transpose_last2_rank3_i8,
    i8,
    kernel_common,
    transpose_last2_rank3_i8
);
#[cfg(feature = "dtype-u32")]
transpose_last2_rank3_fn!(
    transpose_last2_rank3_u32,
    u32,
    kernel_common,
    transpose_last2_rank3_u32
);
#[cfg(feature = "dtype-i32")]
transpose_last2_rank3_fn!(
    transpose_last2_rank3_i32,
    i32,
    kernel_common,
    transpose_last2_rank3_i32
);
#[cfg(feature = "dtype-u64")]
transpose_last2_rank3_fn!(
    transpose_last2_rank3_u64,
    u64,
    kernel_common,
    transpose_last2_rank3_u64
);
#[cfg(feature = "dtype-i64")]
transpose_last2_rank3_fn!(
    transpose_last2_rank3_i64,
    i64,
    kernel_common,
    transpose_last2_rank3_i64
);

macro_rules! transpose_last2_rank4_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            dimensions: [usize; 4],
        ) -> Result<()> {
            let len = checked_rank4_len(dimensions)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let dimensions = checked_rank4_i32(dimensions)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    dimensions[0],
                    dimensions[1],
                    dimensions[2],
                    dimensions[3],
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
transpose_last2_rank4_fn!(
    transpose_last2_rank4_f32,
    f32,
    kernel_f32,
    transpose_last2_rank4_f32
);
#[cfg(feature = "dtype-f16")]
transpose_last2_rank4_fn!(
    transpose_last2_rank4_f16,
    f16,
    kernel_f16,
    transpose_last2_rank4_f16
);
#[cfg(feature = "dtype-f64")]
transpose_last2_rank4_fn!(
    transpose_last2_rank4_f64,
    f64,
    kernel_f64,
    transpose_last2_rank4_f64
);
#[cfg(feature = "dtype-u8")]
transpose_last2_rank4_fn!(
    transpose_last2_rank4_u8,
    u8,
    kernel_common,
    transpose_last2_rank4_u8
);
#[cfg(feature = "dtype-i8")]
transpose_last2_rank4_fn!(
    transpose_last2_rank4_i8,
    i8,
    kernel_common,
    transpose_last2_rank4_i8
);
#[cfg(feature = "dtype-u32")]
transpose_last2_rank4_fn!(
    transpose_last2_rank4_u32,
    u32,
    kernel_common,
    transpose_last2_rank4_u32
);
#[cfg(feature = "dtype-i32")]
transpose_last2_rank4_fn!(
    transpose_last2_rank4_i32,
    i32,
    kernel_common,
    transpose_last2_rank4_i32
);
#[cfg(feature = "dtype-u64")]
transpose_last2_rank4_fn!(
    transpose_last2_rank4_u64,
    u64,
    kernel_common,
    transpose_last2_rank4_u64
);
#[cfg(feature = "dtype-i64")]
transpose_last2_rank4_fn!(
    transpose_last2_rank4_i64,
    i64,
    kernel_common,
    transpose_last2_rank4_i64
);

fn concat_rhs_dimensions<const R: usize>(
    output_dimensions: [usize; R],
    lhs_dimensions: [usize; R],
    axis: usize,
) -> Result<[usize; R]> {
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
    let mut dimensions = output_dimensions;
    dimensions[axis] -= lhs_dimensions[axis];
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

macro_rules! materialize_rank2_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            dimensions: [usize; 2],
            input_strides: [usize; 2],
        ) -> Result<()> {
            let len = crate::utility::checked_element_count(dimensions[0], dimensions[1])?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let dimensions = checked_rank2_i32(dimensions)?;
            let input_strides = checked_rank2_i32(input_strides)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    dimensions[0],
                    dimensions[1],
                    input_strides[0],
                    input_strides[1],
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
materialize_rank2_fn!(
    materialize_rank2_f32,
    f32,
    kernel_f32,
    materialize_rank2_f32
);
#[cfg(feature = "dtype-f16")]
materialize_rank2_fn!(
    materialize_rank2_f16,
    f16,
    kernel_f16,
    materialize_rank2_f16
);
#[cfg(feature = "dtype-f64")]
materialize_rank2_fn!(
    materialize_rank2_f64,
    f64,
    kernel_f64,
    materialize_rank2_f64
);
#[cfg(feature = "dtype-u8")]
materialize_rank2_fn!(
    materialize_rank2_u8,
    u8,
    kernel_common,
    materialize_rank2_u8
);
#[cfg(feature = "dtype-i8")]
materialize_rank2_fn!(
    materialize_rank2_i8,
    i8,
    kernel_common,
    materialize_rank2_i8
);
#[cfg(feature = "dtype-u32")]
materialize_rank2_fn!(
    materialize_rank2_u32,
    u32,
    kernel_common,
    materialize_rank2_u32
);
#[cfg(feature = "dtype-i32")]
materialize_rank2_fn!(
    materialize_rank2_i32,
    i32,
    kernel_common,
    materialize_rank2_i32
);
#[cfg(feature = "dtype-u64")]
materialize_rank2_fn!(
    materialize_rank2_u64,
    u64,
    kernel_common,
    materialize_rank2_u64
);
#[cfg(feature = "dtype-i64")]
materialize_rank2_fn!(
    materialize_rank2_i64,
    i64,
    kernel_common,
    materialize_rank2_i64
);

macro_rules! slice_rank2_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            output_dimensions: [usize; 2],
            input_dimensions: [usize; 2],
            input_strides: [usize; 2],
            starts: [usize; 2],
        ) -> Result<()> {
            ensure_slice_bounds(output_dimensions, input_dimensions, starts)?;
            let len =
                crate::utility::checked_element_count(output_dimensions[0], output_dimensions[1])?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let output_dimensions = checked_rank2_i32(output_dimensions)?;
            let input_strides = checked_rank2_i32(input_strides)?;
            let starts = checked_rank2_i32(starts)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    output_dimensions[0],
                    output_dimensions[1],
                    input_strides[0],
                    input_strides[1],
                    starts[0],
                    starts[1],
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
slice_rank2_fn!(slice_rank2_f32, f32, kernel_f32, slice_rank2_f32);
#[cfg(feature = "dtype-f16")]
slice_rank2_fn!(slice_rank2_f16, f16, kernel_f16, slice_rank2_f16);
#[cfg(feature = "dtype-f64")]
slice_rank2_fn!(slice_rank2_f64, f64, kernel_f64, slice_rank2_f64);
#[cfg(feature = "dtype-u8")]
slice_rank2_fn!(slice_rank2_u8, u8, kernel_common, slice_rank2_u8);
#[cfg(feature = "dtype-i8")]
slice_rank2_fn!(slice_rank2_i8, i8, kernel_common, slice_rank2_i8);
#[cfg(feature = "dtype-u32")]
slice_rank2_fn!(slice_rank2_u32, u32, kernel_common, slice_rank2_u32);
#[cfg(feature = "dtype-i32")]
slice_rank2_fn!(slice_rank2_i32, i32, kernel_common, slice_rank2_i32);
#[cfg(feature = "dtype-u64")]
slice_rank2_fn!(slice_rank2_u64, u64, kernel_common, slice_rank2_u64);
#[cfg(feature = "dtype-i64")]
slice_rank2_fn!(slice_rank2_i64, i64, kernel_common, slice_rank2_i64);

macro_rules! materialize_rank3_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            dimensions: [usize; 3],
            input_strides: [usize; 3],
        ) -> Result<()> {
            let len = crate::utility::checked_element_count(dimensions[0], dimensions[1])?;
            let len = crate::utility::checked_element_count(len, dimensions[2])?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let dimensions = checked_rank3_i32(dimensions)?;
            let input_strides = checked_rank3_i32(input_strides)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    dimensions[0],
                    dimensions[1],
                    dimensions[2],
                    input_strides[0],
                    input_strides[1],
                    input_strides[2],
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
materialize_rank3_fn!(
    materialize_rank3_f32,
    f32,
    kernel_f32,
    materialize_rank3_f32
);
#[cfg(feature = "dtype-f16")]
materialize_rank3_fn!(
    materialize_rank3_f16,
    f16,
    kernel_f16,
    materialize_rank3_f16
);
#[cfg(feature = "dtype-f64")]
materialize_rank3_fn!(
    materialize_rank3_f64,
    f64,
    kernel_f64,
    materialize_rank3_f64
);
#[cfg(feature = "dtype-u8")]
materialize_rank3_fn!(
    materialize_rank3_u8,
    u8,
    kernel_common,
    materialize_rank3_u8
);
#[cfg(feature = "dtype-i8")]
materialize_rank3_fn!(
    materialize_rank3_i8,
    i8,
    kernel_common,
    materialize_rank3_i8
);
#[cfg(feature = "dtype-u32")]
materialize_rank3_fn!(
    materialize_rank3_u32,
    u32,
    kernel_common,
    materialize_rank3_u32
);
#[cfg(feature = "dtype-i32")]
materialize_rank3_fn!(
    materialize_rank3_i32,
    i32,
    kernel_common,
    materialize_rank3_i32
);
#[cfg(feature = "dtype-u64")]
materialize_rank3_fn!(
    materialize_rank3_u64,
    u64,
    kernel_common,
    materialize_rank3_u64
);
#[cfg(feature = "dtype-i64")]
materialize_rank3_fn!(
    materialize_rank3_i64,
    i64,
    kernel_common,
    materialize_rank3_i64
);

macro_rules! slice_rank3_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            output_dimensions: [usize; 3],
            input_dimensions: [usize; 3],
            input_strides: [usize; 3],
            starts: [usize; 3],
        ) -> Result<()> {
            ensure_slice_bounds(output_dimensions, input_dimensions, starts)?;
            let len =
                crate::utility::checked_element_count(output_dimensions[0], output_dimensions[1])?;
            let len = crate::utility::checked_element_count(len, output_dimensions[2])?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let output_dimensions = checked_rank3_i32(output_dimensions)?;
            let input_strides = checked_rank3_i32(input_strides)?;
            let starts = checked_rank3_i32(starts)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    output_dimensions[0],
                    output_dimensions[1],
                    output_dimensions[2],
                    input_strides[0],
                    input_strides[1],
                    input_strides[2],
                    starts[0],
                    starts[1],
                    starts[2],
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
slice_rank3_fn!(slice_rank3_f32, f32, kernel_f32, slice_rank3_f32);
#[cfg(feature = "dtype-f16")]
slice_rank3_fn!(slice_rank3_f16, f16, kernel_f16, slice_rank3_f16);
#[cfg(feature = "dtype-f64")]
slice_rank3_fn!(slice_rank3_f64, f64, kernel_f64, slice_rank3_f64);
#[cfg(feature = "dtype-u8")]
slice_rank3_fn!(slice_rank3_u8, u8, kernel_common, slice_rank3_u8);
#[cfg(feature = "dtype-i8")]
slice_rank3_fn!(slice_rank3_i8, i8, kernel_common, slice_rank3_i8);
#[cfg(feature = "dtype-u32")]
slice_rank3_fn!(slice_rank3_u32, u32, kernel_common, slice_rank3_u32);
#[cfg(feature = "dtype-i32")]
slice_rank3_fn!(slice_rank3_i32, i32, kernel_common, slice_rank3_i32);
#[cfg(feature = "dtype-u64")]
slice_rank3_fn!(slice_rank3_u64, u64, kernel_common, slice_rank3_u64);
#[cfg(feature = "dtype-i64")]
slice_rank3_fn!(slice_rank3_i64, i64, kernel_common, slice_rank3_i64);

macro_rules! pad_rank2_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            output_dimensions: [usize; 2],
            input_dimensions: [usize; 2],
            input_strides: [usize; 2],
            pad_before: [usize; 2],
            pad_value: $ty,
        ) -> Result<()> {
            ensure_pad_bounds(output_dimensions, input_dimensions, pad_before)?;
            let len =
                crate::utility::checked_element_count(output_dimensions[0], output_dimensions[1])?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            let input_len =
                crate::utility::checked_element_count(input_dimensions[0], input_dimensions[1])?;
            if input_len > 0 {
                checked_device_pointer(input)?;
            }
            let output_dimensions = checked_rank2_i32(output_dimensions)?;
            let input_dimensions = checked_rank2_i32(input_dimensions)?;
            let input_strides = checked_rank2_i32(input_strides)?;
            let pad_before = checked_rank2_i32(pad_before)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    output_dimensions[0],
                    output_dimensions[1],
                    input_dimensions[0],
                    input_dimensions[1],
                    input_strides[0],
                    input_strides[1],
                    pad_before[0],
                    pad_before[1],
                    pad_value,
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
pad_rank2_fn!(pad_rank2_f32, f32, kernel_f32, pad_rank2_f32);
#[cfg(feature = "dtype-f16")]
pad_rank2_fn!(pad_rank2_f16, f16, kernel_f16, pad_rank2_f16);
#[cfg(feature = "dtype-f64")]
pad_rank2_fn!(pad_rank2_f64, f64, kernel_f64, pad_rank2_f64);
#[cfg(feature = "dtype-u8")]
pad_rank2_fn!(pad_rank2_u8, u8, kernel_common, pad_rank2_u8);
#[cfg(feature = "dtype-i8")]
pad_rank2_fn!(pad_rank2_i8, i8, kernel_common, pad_rank2_i8);
#[cfg(feature = "dtype-u32")]
pad_rank2_fn!(pad_rank2_u32, u32, kernel_common, pad_rank2_u32);
#[cfg(feature = "dtype-i32")]
pad_rank2_fn!(pad_rank2_i32, i32, kernel_common, pad_rank2_i32);
#[cfg(feature = "dtype-u64")]
pad_rank2_fn!(pad_rank2_u64, u64, kernel_common, pad_rank2_u64);
#[cfg(feature = "dtype-i64")]
pad_rank2_fn!(pad_rank2_i64, i64, kernel_common, pad_rank2_i64);

macro_rules! pad_rank3_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            output_dimensions: [usize; 3],
            input_dimensions: [usize; 3],
            input_strides: [usize; 3],
            pad_before: [usize; 3],
            pad_value: $ty,
        ) -> Result<()> {
            ensure_pad_bounds(output_dimensions, input_dimensions, pad_before)?;
            let len =
                crate::utility::checked_element_count(output_dimensions[0], output_dimensions[1])?;
            let len = crate::utility::checked_element_count(len, output_dimensions[2])?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            let input_len =
                crate::utility::checked_element_count(input_dimensions[0], input_dimensions[1])?;
            let input_len = crate::utility::checked_element_count(input_len, input_dimensions[2])?;
            if input_len > 0 {
                checked_device_pointer(input)?;
            }
            let output_dimensions = checked_rank3_i32(output_dimensions)?;
            let input_dimensions = checked_rank3_i32(input_dimensions)?;
            let input_strides = checked_rank3_i32(input_strides)?;
            let pad_before = checked_rank3_i32(pad_before)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    output_dimensions[0],
                    output_dimensions[1],
                    output_dimensions[2],
                    input_dimensions[0],
                    input_dimensions[1],
                    input_dimensions[2],
                    input_strides[0],
                    input_strides[1],
                    input_strides[2],
                    pad_before[0],
                    pad_before[1],
                    pad_before[2],
                    pad_value,
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
pad_rank3_fn!(pad_rank3_f32, f32, kernel_f32, pad_rank3_f32);
#[cfg(feature = "dtype-f16")]
pad_rank3_fn!(pad_rank3_f16, f16, kernel_f16, pad_rank3_f16);
#[cfg(feature = "dtype-f64")]
pad_rank3_fn!(pad_rank3_f64, f64, kernel_f64, pad_rank3_f64);
#[cfg(feature = "dtype-u8")]
pad_rank3_fn!(pad_rank3_u8, u8, kernel_common, pad_rank3_u8);
#[cfg(feature = "dtype-i8")]
pad_rank3_fn!(pad_rank3_i8, i8, kernel_common, pad_rank3_i8);
#[cfg(feature = "dtype-u32")]
pad_rank3_fn!(pad_rank3_u32, u32, kernel_common, pad_rank3_u32);
#[cfg(feature = "dtype-i32")]
pad_rank3_fn!(pad_rank3_i32, i32, kernel_common, pad_rank3_i32);
#[cfg(feature = "dtype-u64")]
pad_rank3_fn!(pad_rank3_u64, u64, kernel_common, pad_rank3_u64);
#[cfg(feature = "dtype-i64")]
pad_rank3_fn!(pad_rank3_i64, i64, kernel_common, pad_rank3_i64);

macro_rules! materialize_rank4_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            dimensions: [usize; 4],
            input_strides: [usize; 4],
        ) -> Result<()> {
            let len = checked_rank4_len(dimensions)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let dimensions = checked_rank4_i32(dimensions)?;
            let input_strides = checked_rank4_i32(input_strides)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    dimensions[0],
                    dimensions[1],
                    dimensions[2],
                    dimensions[3],
                    input_strides[0],
                    input_strides[1],
                    input_strides[2],
                    input_strides[3],
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
materialize_rank4_fn!(
    materialize_rank4_f32,
    f32,
    kernel_f32,
    materialize_rank4_f32
);
#[cfg(feature = "dtype-f16")]
materialize_rank4_fn!(
    materialize_rank4_f16,
    f16,
    kernel_f16,
    materialize_rank4_f16
);
#[cfg(feature = "dtype-f64")]
materialize_rank4_fn!(
    materialize_rank4_f64,
    f64,
    kernel_f64,
    materialize_rank4_f64
);
#[cfg(feature = "dtype-u8")]
materialize_rank4_fn!(
    materialize_rank4_u8,
    u8,
    kernel_common,
    materialize_rank4_u8
);
#[cfg(feature = "dtype-i8")]
materialize_rank4_fn!(
    materialize_rank4_i8,
    i8,
    kernel_common,
    materialize_rank4_i8
);
#[cfg(feature = "dtype-u32")]
materialize_rank4_fn!(
    materialize_rank4_u32,
    u32,
    kernel_common,
    materialize_rank4_u32
);
#[cfg(feature = "dtype-i32")]
materialize_rank4_fn!(
    materialize_rank4_i32,
    i32,
    kernel_common,
    materialize_rank4_i32
);
#[cfg(feature = "dtype-u64")]
materialize_rank4_fn!(
    materialize_rank4_u64,
    u64,
    kernel_common,
    materialize_rank4_u64
);
#[cfg(feature = "dtype-i64")]
materialize_rank4_fn!(
    materialize_rank4_i64,
    i64,
    kernel_common,
    materialize_rank4_i64
);

macro_rules! pad_rank4_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            output_dimensions: [usize; 4],
            input_dimensions: [usize; 4],
            input_strides: [usize; 4],
            pad_before: [usize; 4],
            pad_value: $ty,
        ) -> Result<()> {
            ensure_pad_bounds(output_dimensions, input_dimensions, pad_before)?;
            let len = checked_rank4_len(output_dimensions)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            let input_len = checked_rank4_len(input_dimensions)?;
            if input_len > 0 {
                checked_device_pointer(input)?;
            }
            let output_dimensions = checked_rank4_i32(output_dimensions)?;
            let input_dimensions = checked_rank4_i32(input_dimensions)?;
            let input_strides = checked_rank4_i32(input_strides)?;
            let pad_before = checked_rank4_i32(pad_before)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    output_dimensions[0],
                    output_dimensions[1],
                    output_dimensions[2],
                    output_dimensions[3],
                    input_dimensions[0],
                    input_dimensions[1],
                    input_dimensions[2],
                    input_dimensions[3],
                    input_strides[0],
                    input_strides[1],
                    input_strides[2],
                    input_strides[3],
                    pad_before[0],
                    pad_before[1],
                    pad_before[2],
                    pad_before[3],
                    pad_value,
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
pad_rank4_fn!(pad_rank4_f32, f32, kernel_f32, pad_rank4_f32);
#[cfg(feature = "dtype-f16")]
pad_rank4_fn!(pad_rank4_f16, f16, kernel_f16, pad_rank4_f16);
#[cfg(feature = "dtype-f64")]
pad_rank4_fn!(pad_rank4_f64, f64, kernel_f64, pad_rank4_f64);
#[cfg(feature = "dtype-u8")]
pad_rank4_fn!(pad_rank4_u8, u8, kernel_common, pad_rank4_u8);
#[cfg(feature = "dtype-i8")]
pad_rank4_fn!(pad_rank4_i8, i8, kernel_common, pad_rank4_i8);
#[cfg(feature = "dtype-u32")]
pad_rank4_fn!(pad_rank4_u32, u32, kernel_common, pad_rank4_u32);
#[cfg(feature = "dtype-i32")]
pad_rank4_fn!(pad_rank4_i32, i32, kernel_common, pad_rank4_i32);
#[cfg(feature = "dtype-u64")]
pad_rank4_fn!(pad_rank4_u64, u64, kernel_common, pad_rank4_u64);
#[cfg(feature = "dtype-i64")]
pad_rank4_fn!(pad_rank4_i64, i64, kernel_common, pad_rank4_i64);

macro_rules! slice_rank4_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            output_dimensions: [usize; 4],
            input_dimensions: [usize; 4],
            input_strides: [usize; 4],
            starts: [usize; 4],
        ) -> Result<()> {
            ensure_slice_bounds(output_dimensions, input_dimensions, starts)?;
            let len = checked_rank4_len(output_dimensions)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let output_dimensions = checked_rank4_i32(output_dimensions)?;
            let input_strides = checked_rank4_i32(input_strides)?;
            let starts = checked_rank4_i32(starts)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    output_dimensions[0],
                    output_dimensions[1],
                    output_dimensions[2],
                    output_dimensions[3],
                    input_strides[0],
                    input_strides[1],
                    input_strides[2],
                    input_strides[3],
                    starts[0],
                    starts[1],
                    starts[2],
                    starts[3],
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
slice_rank4_fn!(slice_rank4_f32, f32, kernel_f32, slice_rank4_f32);
#[cfg(feature = "dtype-f16")]
slice_rank4_fn!(slice_rank4_f16, f16, kernel_f16, slice_rank4_f16);
#[cfg(feature = "dtype-f64")]
slice_rank4_fn!(slice_rank4_f64, f64, kernel_f64, slice_rank4_f64);
#[cfg(feature = "dtype-u8")]
slice_rank4_fn!(slice_rank4_u8, u8, kernel_common, slice_rank4_u8);
#[cfg(feature = "dtype-i8")]
slice_rank4_fn!(slice_rank4_i8, i8, kernel_common, slice_rank4_i8);
#[cfg(feature = "dtype-u32")]
slice_rank4_fn!(slice_rank4_u32, u32, kernel_common, slice_rank4_u32);
#[cfg(feature = "dtype-i32")]
slice_rank4_fn!(slice_rank4_i32, i32, kernel_common, slice_rank4_i32);
#[cfg(feature = "dtype-u64")]
slice_rank4_fn!(slice_rank4_u64, u64, kernel_common, slice_rank4_u64);
#[cfg(feature = "dtype-i64")]
slice_rank4_fn!(slice_rank4_i64, i64, kernel_common, slice_rank4_i64);

macro_rules! concat_rank2_fn {
    (
        $name:ident,
        $ty:ty,
        $kernel:ident,
        $axis0:ident,
        $axis1:ident
    ) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lhs: DevicePointer<$ty>,
            rhs: DevicePointer<$ty>,
            output_dimensions: [usize; 2],
            lhs_dimensions: [usize; 2],
            lhs_strides: [usize; 2],
            rhs_strides: [usize; 2],
            axis: usize,
        ) -> Result<()> {
            let rhs_dimensions = concat_rhs_dimensions(output_dimensions, lhs_dimensions, axis)?;
            let len =
                crate::utility::checked_element_count(output_dimensions[0], output_dimensions[1])?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            if crate::utility::checked_element_count(lhs_dimensions[0], lhs_dimensions[1])? > 0 {
                checked_device_pointer(lhs)?;
            }
            if crate::utility::checked_element_count(rhs_dimensions[0], rhs_dimensions[1])? > 0 {
                checked_device_pointer(rhs)?;
            }
            let output_dimensions = checked_rank2_i32(output_dimensions)?;
            let lhs_dimensions = checked_rank2_i32(lhs_dimensions)?;
            let lhs_strides = checked_rank2_i32(lhs_strides)?;
            let rhs_strides = checked_rank2_i32(rhs_strides)?;
            let launch = VectorLaunch::create(len)?;
            macro_rules! enqueue {
                ($kernel_fn:ident) => {
                    unsafe {
                        $kernel::$kernel_fn(
                            out,
                            lhs,
                            rhs,
                            output_dimensions[0],
                            output_dimensions[1],
                            lhs_dimensions[0],
                            lhs_dimensions[1],
                            lhs_strides[0],
                            lhs_strides[1],
                            rhs_strides[0],
                            rhs_strides[1],
                            launch.len_i32,
                        )
                    }
                    .grid(launch.grid)
                    .enqueue_on(stream)
                };
            }
            match axis {
                0 => enqueue!($axis0),
                1 => enqueue!($axis1),
                _ => {
                    return Err(Error::UnsupportedAxis {
                        op: "concat".into(),
                        axis,
                    })
                }
            }?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
concat_rank2_fn!(
    concat_rank2_f32,
    f32,
    kernel_f32,
    concat_rank2_f32_axis0,
    concat_rank2_f32_axis1
);
#[cfg(feature = "dtype-f16")]
concat_rank2_fn!(
    concat_rank2_f16,
    f16,
    kernel_f16,
    concat_rank2_f16_axis0,
    concat_rank2_f16_axis1
);
#[cfg(feature = "dtype-f64")]
concat_rank2_fn!(
    concat_rank2_f64,
    f64,
    kernel_f64,
    concat_rank2_f64_axis0,
    concat_rank2_f64_axis1
);
#[cfg(feature = "dtype-u8")]
concat_rank2_fn!(
    concat_rank2_u8,
    u8,
    kernel_common,
    concat_rank2_u8_axis0,
    concat_rank2_u8_axis1
);
#[cfg(feature = "dtype-i8")]
concat_rank2_fn!(
    concat_rank2_i8,
    i8,
    kernel_common,
    concat_rank2_i8_axis0,
    concat_rank2_i8_axis1
);
#[cfg(feature = "dtype-u32")]
concat_rank2_fn!(
    concat_rank2_u32,
    u32,
    kernel_common,
    concat_rank2_u32_axis0,
    concat_rank2_u32_axis1
);
#[cfg(feature = "dtype-i32")]
concat_rank2_fn!(
    concat_rank2_i32,
    i32,
    kernel_common,
    concat_rank2_i32_axis0,
    concat_rank2_i32_axis1
);
#[cfg(feature = "dtype-u64")]
concat_rank2_fn!(
    concat_rank2_u64,
    u64,
    kernel_common,
    concat_rank2_u64_axis0,
    concat_rank2_u64_axis1
);
#[cfg(feature = "dtype-i64")]
concat_rank2_fn!(
    concat_rank2_i64,
    i64,
    kernel_common,
    concat_rank2_i64_axis0,
    concat_rank2_i64_axis1
);

macro_rules! concat_rank3_fn {
    (
        $name:ident,
        $ty:ty,
        $kernel:ident,
        $axis0:ident,
        $axis1:ident,
        $axis2:ident
    ) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lhs: DevicePointer<$ty>,
            rhs: DevicePointer<$ty>,
            output_dimensions: [usize; 3],
            lhs_dimensions: [usize; 3],
            lhs_strides: [usize; 3],
            rhs_strides: [usize; 3],
            axis: usize,
        ) -> Result<()> {
            let rhs_dimensions = concat_rhs_dimensions(output_dimensions, lhs_dimensions, axis)?;
            let len =
                crate::utility::checked_element_count(output_dimensions[0], output_dimensions[1])?;
            let len = crate::utility::checked_element_count(len, output_dimensions[2])?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            let lhs_len =
                crate::utility::checked_element_count(lhs_dimensions[0], lhs_dimensions[1])?;
            let lhs_len = crate::utility::checked_element_count(lhs_len, lhs_dimensions[2])?;
            if lhs_len > 0 {
                checked_device_pointer(lhs)?;
            }
            let rhs_len =
                crate::utility::checked_element_count(rhs_dimensions[0], rhs_dimensions[1])?;
            let rhs_len = crate::utility::checked_element_count(rhs_len, rhs_dimensions[2])?;
            if rhs_len > 0 {
                checked_device_pointer(rhs)?;
            }
            let output_dimensions = checked_rank3_i32(output_dimensions)?;
            let lhs_dimensions = checked_rank3_i32(lhs_dimensions)?;
            let lhs_strides = checked_rank3_i32(lhs_strides)?;
            let rhs_strides = checked_rank3_i32(rhs_strides)?;
            let launch = VectorLaunch::create(len)?;
            macro_rules! enqueue {
                ($kernel_fn:ident) => {
                    unsafe {
                        $kernel::$kernel_fn(
                            out,
                            lhs,
                            rhs,
                            output_dimensions[0],
                            output_dimensions[1],
                            output_dimensions[2],
                            lhs_dimensions[0],
                            lhs_dimensions[1],
                            lhs_dimensions[2],
                            lhs_strides[0],
                            lhs_strides[1],
                            lhs_strides[2],
                            rhs_strides[0],
                            rhs_strides[1],
                            rhs_strides[2],
                            launch.len_i32,
                        )
                    }
                    .grid(launch.grid)
                    .enqueue_on(stream)
                };
            }
            match axis {
                0 => enqueue!($axis0),
                1 => enqueue!($axis1),
                2 => enqueue!($axis2),
                _ => {
                    return Err(Error::UnsupportedAxis {
                        op: "concat".into(),
                        axis,
                    })
                }
            }?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
concat_rank3_fn!(
    concat_rank3_f32,
    f32,
    kernel_f32,
    concat_rank3_f32_axis0,
    concat_rank3_f32_axis1,
    concat_rank3_f32_axis2
);
#[cfg(feature = "dtype-f16")]
concat_rank3_fn!(
    concat_rank3_f16,
    f16,
    kernel_f16,
    concat_rank3_f16_axis0,
    concat_rank3_f16_axis1,
    concat_rank3_f16_axis2
);
#[cfg(feature = "dtype-f64")]
concat_rank3_fn!(
    concat_rank3_f64,
    f64,
    kernel_f64,
    concat_rank3_f64_axis0,
    concat_rank3_f64_axis1,
    concat_rank3_f64_axis2
);
#[cfg(feature = "dtype-u8")]
concat_rank3_fn!(
    concat_rank3_u8,
    u8,
    kernel_common,
    concat_rank3_u8_axis0,
    concat_rank3_u8_axis1,
    concat_rank3_u8_axis2
);
#[cfg(feature = "dtype-i8")]
concat_rank3_fn!(
    concat_rank3_i8,
    i8,
    kernel_common,
    concat_rank3_i8_axis0,
    concat_rank3_i8_axis1,
    concat_rank3_i8_axis2
);
#[cfg(feature = "dtype-u32")]
concat_rank3_fn!(
    concat_rank3_u32,
    u32,
    kernel_common,
    concat_rank3_u32_axis0,
    concat_rank3_u32_axis1,
    concat_rank3_u32_axis2
);
#[cfg(feature = "dtype-i32")]
concat_rank3_fn!(
    concat_rank3_i32,
    i32,
    kernel_common,
    concat_rank3_i32_axis0,
    concat_rank3_i32_axis1,
    concat_rank3_i32_axis2
);
#[cfg(feature = "dtype-u64")]
concat_rank3_fn!(
    concat_rank3_u64,
    u64,
    kernel_common,
    concat_rank3_u64_axis0,
    concat_rank3_u64_axis1,
    concat_rank3_u64_axis2
);
#[cfg(feature = "dtype-i64")]
concat_rank3_fn!(
    concat_rank3_i64,
    i64,
    kernel_common,
    concat_rank3_i64_axis0,
    concat_rank3_i64_axis1,
    concat_rank3_i64_axis2
);

macro_rules! concat_rank4_fn {
    (
        $name:ident,
        $ty:ty,
        $kernel:ident,
        $axis0:ident,
        $axis1:ident,
        $axis2:ident,
        $axis3:ident
    ) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            lhs: DevicePointer<$ty>,
            rhs: DevicePointer<$ty>,
            output_dimensions: [usize; 4],
            lhs_dimensions: [usize; 4],
            lhs_strides: [usize; 4],
            rhs_strides: [usize; 4],
            axis: usize,
        ) -> Result<()> {
            let rhs_dimensions = concat_rhs_dimensions(output_dimensions, lhs_dimensions, axis)?;
            let len = checked_rank4_len(output_dimensions)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            if checked_rank4_len(lhs_dimensions)? > 0 {
                checked_device_pointer(lhs)?;
            }
            if checked_rank4_len(rhs_dimensions)? > 0 {
                checked_device_pointer(rhs)?;
            }
            let output_dimensions = checked_rank4_i32(output_dimensions)?;
            let lhs_dimensions = checked_rank4_i32(lhs_dimensions)?;
            let lhs_strides = checked_rank4_i32(lhs_strides)?;
            let rhs_strides = checked_rank4_i32(rhs_strides)?;
            let launch = VectorLaunch::create(len)?;
            macro_rules! enqueue {
                ($kernel_fn:ident) => {
                    unsafe {
                        $kernel::$kernel_fn(
                            out,
                            lhs,
                            rhs,
                            output_dimensions[0],
                            output_dimensions[1],
                            output_dimensions[2],
                            output_dimensions[3],
                            lhs_dimensions[0],
                            lhs_dimensions[1],
                            lhs_dimensions[2],
                            lhs_dimensions[3],
                            lhs_strides[0],
                            lhs_strides[1],
                            lhs_strides[2],
                            lhs_strides[3],
                            rhs_strides[0],
                            rhs_strides[1],
                            rhs_strides[2],
                            rhs_strides[3],
                            launch.len_i32,
                        )
                    }
                    .grid(launch.grid)
                    .enqueue_on(stream)
                };
            }
            match axis {
                0 => enqueue!($axis0),
                1 => enqueue!($axis1),
                2 => enqueue!($axis2),
                3 => enqueue!($axis3),
                _ => {
                    return Err(Error::UnsupportedAxis {
                        op: "concat".into(),
                        axis,
                    })
                }
            }?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
concat_rank4_fn!(
    concat_rank4_f32,
    f32,
    kernel_f32,
    concat_rank4_f32_axis0,
    concat_rank4_f32_axis1,
    concat_rank4_f32_axis2,
    concat_rank4_f32_axis3
);
#[cfg(feature = "dtype-f16")]
concat_rank4_fn!(
    concat_rank4_f16,
    f16,
    kernel_f16,
    concat_rank4_f16_axis0,
    concat_rank4_f16_axis1,
    concat_rank4_f16_axis2,
    concat_rank4_f16_axis3
);
#[cfg(feature = "dtype-f64")]
concat_rank4_fn!(
    concat_rank4_f64,
    f64,
    kernel_f64,
    concat_rank4_f64_axis0,
    concat_rank4_f64_axis1,
    concat_rank4_f64_axis2,
    concat_rank4_f64_axis3
);
#[cfg(feature = "dtype-u8")]
concat_rank4_fn!(
    concat_rank4_u8,
    u8,
    kernel_common,
    concat_rank4_u8_axis0,
    concat_rank4_u8_axis1,
    concat_rank4_u8_axis2,
    concat_rank4_u8_axis3
);
#[cfg(feature = "dtype-i8")]
concat_rank4_fn!(
    concat_rank4_i8,
    i8,
    kernel_common,
    concat_rank4_i8_axis0,
    concat_rank4_i8_axis1,
    concat_rank4_i8_axis2,
    concat_rank4_i8_axis3
);
#[cfg(feature = "dtype-u32")]
concat_rank4_fn!(
    concat_rank4_u32,
    u32,
    kernel_common,
    concat_rank4_u32_axis0,
    concat_rank4_u32_axis1,
    concat_rank4_u32_axis2,
    concat_rank4_u32_axis3
);
#[cfg(feature = "dtype-i32")]
concat_rank4_fn!(
    concat_rank4_i32,
    i32,
    kernel_common,
    concat_rank4_i32_axis0,
    concat_rank4_i32_axis1,
    concat_rank4_i32_axis2,
    concat_rank4_i32_axis3
);
#[cfg(feature = "dtype-u64")]
concat_rank4_fn!(
    concat_rank4_u64,
    u64,
    kernel_common,
    concat_rank4_u64_axis0,
    concat_rank4_u64_axis1,
    concat_rank4_u64_axis2,
    concat_rank4_u64_axis3
);
#[cfg(feature = "dtype-i64")]
concat_rank4_fn!(
    concat_rank4_i64,
    i64,
    kernel_common,
    concat_rank4_i64_axis0,
    concat_rank4_i64_axis1,
    concat_rank4_i64_axis2,
    concat_rank4_i64_axis3
);

macro_rules! repeat_kv_rank4_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            output_dimensions: [usize; 4],
            input_strides: [usize; 4],
            repeats: usize,
        ) -> Result<()> {
            let input_dimensions = repeat_kv_input_dimensions(output_dimensions, repeats)?;
            let len = checked_rank4_len(output_dimensions)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            if checked_rank4_len(input_dimensions)? > 0 {
                checked_device_pointer(input)?;
            }
            let output_dimensions = checked_rank4_i32(output_dimensions)?;
            let input_strides = checked_rank4_i32(input_strides)?;
            let repeats = checked_i32_value(repeats)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    output_dimensions[0],
                    output_dimensions[1],
                    output_dimensions[2],
                    output_dimensions[3],
                    input_strides[0],
                    input_strides[1],
                    input_strides[2],
                    input_strides[3],
                    repeats,
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
repeat_kv_rank4_fn!(repeat_kv_rank4_f32, f32, kernel_f32, repeat_kv_rank4_f32);
#[cfg(feature = "dtype-f16")]
repeat_kv_rank4_fn!(repeat_kv_rank4_f16, f16, kernel_f16, repeat_kv_rank4_f16);
#[cfg(feature = "dtype-f64")]
repeat_kv_rank4_fn!(repeat_kv_rank4_f64, f64, kernel_f64, repeat_kv_rank4_f64);
#[cfg(feature = "dtype-u8")]
repeat_kv_rank4_fn!(repeat_kv_rank4_u8, u8, kernel_common, repeat_kv_rank4_u8);
#[cfg(feature = "dtype-i8")]
repeat_kv_rank4_fn!(repeat_kv_rank4_i8, i8, kernel_common, repeat_kv_rank4_i8);
#[cfg(feature = "dtype-u32")]
repeat_kv_rank4_fn!(repeat_kv_rank4_u32, u32, kernel_common, repeat_kv_rank4_u32);
#[cfg(feature = "dtype-i32")]
repeat_kv_rank4_fn!(repeat_kv_rank4_i32, i32, kernel_common, repeat_kv_rank4_i32);
#[cfg(feature = "dtype-u64")]
repeat_kv_rank4_fn!(repeat_kv_rank4_u64, u64, kernel_common, repeat_kv_rank4_u64);
#[cfg(feature = "dtype-i64")]
repeat_kv_rank4_fn!(repeat_kv_rank4_i64, i64, kernel_common, repeat_kv_rank4_i64);

#[cfg(feature = "dtype-f32")]
pub fn pack_downsample_2d_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    rows: usize,
    cols: usize,
    factor: usize,
) -> Result<()> {
    if rows == 0 || cols == 0 || factor == 0 {
        return Err(Error::InvalidLength);
    }
    if !rows.is_multiple_of(factor) {
        return Err(Error::LengthMismatch);
    }
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    let len = crate::utility::checked_element_count(rows, cols)?;
    let launch = VectorLaunch::create(len)?;
    let rows = checked_i32_value(rows)?;
    let cols = checked_i32_value(cols)?;
    let factor = checked_i32_value(factor)?;
    unsafe { kernel_f32::pack_downsample_2d_f32(out, input, rows, cols, factor, launch.len_i32) }
        .grid(launch.grid)
        .enqueue_on(stream)?;
    Ok(())
}
