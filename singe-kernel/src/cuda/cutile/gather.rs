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
use crate::cuda::cutile::kernel::f16::gather as kernel_f16;
#[cfg(feature = "dtype-f32")]
use crate::cuda::cutile::kernel::f32::gather as kernel_f32;
#[cfg(feature = "dtype-f64")]
use crate::cuda::cutile::kernel::f64::gather as kernel_f64;
use crate::{
    cuda::cutile::{
        DeviceOpExt,
        utility::{VectorLaunch, checked_device_pointer, raw_vector_grid},
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

macro_rules! gather_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            indices: DevicePointer<u32>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(indices)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { $kernel::$kernel_fn(out, input, indices, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
gather_fn!(gather_f16, f16, kernel_f16, gather_f16);
#[cfg(feature = "dtype-f32")]
gather_fn!(gather_f32, f32, kernel_f32, gather_f32);
#[cfg(feature = "dtype-f64")]
gather_fn!(gather_f64, f64, kernel_f64, gather_f64);
#[cfg(feature = "dtype-u8")]
gather_fn!(gather_u8, u8, kernel_common, gather_u8);
#[cfg(feature = "dtype-i8")]
gather_fn!(gather_i8, i8, kernel_common, gather_i8);
#[cfg(feature = "dtype-u32")]
gather_fn!(gather_u32, u32, kernel_common, gather_u32);
#[cfg(feature = "dtype-i32")]
gather_fn!(gather_i32, i32, kernel_common, gather_i32);
#[cfg(feature = "dtype-u64")]
gather_fn!(gather_u64, u64, kernel_common, gather_u64);
#[cfg(feature = "dtype-i64")]
gather_fn!(gather_i64, i64, kernel_common, gather_i64);

macro_rules! scatter_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            indices: DevicePointer<u32>,
            len: usize,
        ) -> Result<()> {
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(indices)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { $kernel::$kernel_fn(out, input, indices, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
scatter_fn!(scatter_f16, f16, kernel_f16, scatter_f16);
#[cfg(feature = "dtype-f32")]
scatter_fn!(scatter_f32, f32, kernel_f32, scatter_f32);
#[cfg(feature = "dtype-f64")]
scatter_fn!(scatter_f64, f64, kernel_f64, scatter_f64);
#[cfg(feature = "dtype-u8")]
scatter_fn!(scatter_u8, u8, kernel_common, scatter_u8);
#[cfg(feature = "dtype-i8")]
scatter_fn!(scatter_i8, i8, kernel_common, scatter_i8);
#[cfg(feature = "dtype-u32")]
scatter_fn!(scatter_u32, u32, kernel_common, scatter_u32);
#[cfg(feature = "dtype-i32")]
scatter_fn!(scatter_i32, i32, kernel_common, scatter_i32);
#[cfg(feature = "dtype-u64")]
scatter_fn!(scatter_u64, u64, kernel_common, scatter_u64);
#[cfg(feature = "dtype-i64")]
scatter_fn!(scatter_i64, i64, kernel_common, scatter_i64);

macro_rules! gather_rows_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            row_indices: DevicePointer<u32>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(row_indices)?;
            let cols = checked_i32_value(cols)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { $kernel::$kernel_fn(out, input, row_indices, cols, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
gather_rows_fn!(gather_rows_f16, f16, kernel_f16, gather_rows_f16);
#[cfg(feature = "dtype-f32")]
gather_rows_fn!(gather_rows_f32, f32, kernel_f32, gather_rows_f32);
#[cfg(feature = "dtype-f64")]
gather_rows_fn!(gather_rows_f64, f64, kernel_f64, gather_rows_f64);
#[cfg(feature = "dtype-u8")]
gather_rows_fn!(gather_rows_u8, u8, kernel_common, gather_rows_u8);
#[cfg(feature = "dtype-i8")]
gather_rows_fn!(gather_rows_i8, i8, kernel_common, gather_rows_i8);
#[cfg(feature = "dtype-u32")]
gather_rows_fn!(gather_rows_u32, u32, kernel_common, gather_rows_u32);
#[cfg(feature = "dtype-i32")]
gather_rows_fn!(gather_rows_i32, i32, kernel_common, gather_rows_i32);
#[cfg(feature = "dtype-u64")]
gather_rows_fn!(gather_rows_u64, u64, kernel_common, gather_rows_u64);
#[cfg(feature = "dtype-i64")]
gather_rows_fn!(gather_rows_i64, i64, kernel_common, gather_rows_i64);

macro_rules! gather_row_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            row_index: usize,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            if row_index >= rows {
                return Err(Error::InvalidLength);
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            let row_index = checked_i32_value(row_index)?;
            let cols_i32 = checked_i32_value(cols)?;
            let grid = raw_vector_grid(cols)?;
            unsafe { $kernel::$kernel_fn(out, input, row_index, cols_i32) }
                .grid(grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
gather_row_fn!(gather_row_f16, f16, kernel_f16, gather_row_f16);
#[cfg(feature = "dtype-f32")]
gather_row_fn!(gather_row_f32, f32, kernel_f32, gather_row_f32);
#[cfg(feature = "dtype-f64")]
gather_row_fn!(gather_row_f64, f64, kernel_f64, gather_row_f64);
#[cfg(feature = "dtype-u8")]
gather_row_fn!(gather_row_u8, u8, kernel_common, gather_row_u8);
#[cfg(feature = "dtype-i8")]
gather_row_fn!(gather_row_i8, i8, kernel_common, gather_row_i8);
#[cfg(feature = "dtype-u32")]
gather_row_fn!(gather_row_u32, u32, kernel_common, gather_row_u32);
#[cfg(feature = "dtype-i32")]
gather_row_fn!(gather_row_i32, i32, kernel_common, gather_row_i32);
#[cfg(feature = "dtype-u64")]
gather_row_fn!(gather_row_u64, u64, kernel_common, gather_row_u64);
#[cfg(feature = "dtype-i64")]
gather_row_fn!(gather_row_i64, i64, kernel_common, gather_row_i64);

macro_rules! scatter_rows_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            row_indices: DevicePointer<u32>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(row_indices)?;
            let cols = checked_i32_value(cols)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { $kernel::$kernel_fn(out, input, row_indices, cols, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
scatter_rows_fn!(scatter_rows_f16, f16, kernel_f16, scatter_rows_f16);
#[cfg(feature = "dtype-f32")]
scatter_rows_fn!(scatter_rows_f32, f32, kernel_f32, scatter_rows_f32);
#[cfg(feature = "dtype-f64")]
scatter_rows_fn!(scatter_rows_f64, f64, kernel_f64, scatter_rows_f64);
#[cfg(feature = "dtype-u8")]
scatter_rows_fn!(scatter_rows_u8, u8, kernel_common, scatter_rows_u8);
#[cfg(feature = "dtype-i8")]
scatter_rows_fn!(scatter_rows_i8, i8, kernel_common, scatter_rows_i8);
#[cfg(feature = "dtype-u32")]
scatter_rows_fn!(scatter_rows_u32, u32, kernel_common, scatter_rows_u32);
#[cfg(feature = "dtype-i32")]
scatter_rows_fn!(scatter_rows_i32, i32, kernel_common, scatter_rows_i32);
#[cfg(feature = "dtype-u64")]
scatter_rows_fn!(scatter_rows_u64, u64, kernel_common, scatter_rows_u64);
#[cfg(feature = "dtype-i64")]
scatter_rows_fn!(scatter_rows_i64, i64, kernel_common, scatter_rows_i64);

macro_rules! copy_indexed_rows_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            source_indices: DevicePointer<u32>,
            output_indices: DevicePointer<u32>,
            rows: usize,
            cols: usize,
        ) -> Result<()> {
            if cols == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(rows, cols)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(source_indices)?;
            checked_device_pointer(output_indices)?;
            let cols = checked_i32_value(cols)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    source_indices,
                    output_indices,
                    cols,
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
copy_indexed_rows_fn!(
    copy_indexed_rows_f16,
    f16,
    kernel_f16,
    copy_indexed_rows_f16
);
#[cfg(feature = "dtype-f32")]
copy_indexed_rows_fn!(
    copy_indexed_rows_f32,
    f32,
    kernel_f32,
    copy_indexed_rows_f32
);
#[cfg(feature = "dtype-f64")]
copy_indexed_rows_fn!(
    copy_indexed_rows_f64,
    f64,
    kernel_f64,
    copy_indexed_rows_f64
);
#[cfg(feature = "dtype-u8")]
copy_indexed_rows_fn!(
    copy_indexed_rows_u8,
    u8,
    kernel_common,
    copy_indexed_rows_u8
);
#[cfg(feature = "dtype-i8")]
copy_indexed_rows_fn!(
    copy_indexed_rows_i8,
    i8,
    kernel_common,
    copy_indexed_rows_i8
);
#[cfg(feature = "dtype-u32")]
copy_indexed_rows_fn!(
    copy_indexed_rows_u32,
    u32,
    kernel_common,
    copy_indexed_rows_u32
);
#[cfg(feature = "dtype-i32")]
copy_indexed_rows_fn!(
    copy_indexed_rows_i32,
    i32,
    kernel_common,
    copy_indexed_rows_i32
);
#[cfg(feature = "dtype-u64")]
copy_indexed_rows_fn!(
    copy_indexed_rows_u64,
    u64,
    kernel_common,
    copy_indexed_rows_u64
);
#[cfg(feature = "dtype-i64")]
copy_indexed_rows_fn!(
    copy_indexed_rows_i64,
    i64,
    kernel_common,
    copy_indexed_rows_i64
);
