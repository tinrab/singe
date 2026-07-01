use std::sync::Arc;

#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

#[cfg(feature = "dtype-f16")]
use crate::cuda::cutile::kernel::f16::embedding as kernel_f16;
#[cfg(feature = "dtype-f32")]
use crate::cuda::cutile::kernel::f32::embedding as kernel_f32;
#[cfg(feature = "dtype-f64")]
use crate::cuda::cutile::kernel::f64::embedding as kernel_f64;
use crate::{
    cuda::cutile::{
        DeviceOpExt,
        utility::{VectorLaunch, checked_device_pointer},
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

macro_rules! embedding_lookup_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            token_ids: DevicePointer<u32>,
            table: DevicePointer<$ty>,
            token_count: usize,
            width: usize,
        ) -> Result<()> {
            if width == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(token_count, width)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(token_ids)?;
            checked_device_pointer(table)?;
            let width = checked_i32_value(width)?;
            let launch = VectorLaunch::create(len)?;
            unsafe { $kernel::$kernel_fn(out, token_ids, table, width, launch.len_i32) }
                .grid(launch.grid)
                .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f16")]
embedding_lookup_fn!(embedding_lookup_f16, f16, kernel_f16, embedding_lookup_f16);
#[cfg(feature = "dtype-f32")]
embedding_lookup_fn!(embedding_lookup_f32, f32, kernel_f32, embedding_lookup_f32);
#[cfg(feature = "dtype-f64")]
embedding_lookup_fn!(embedding_lookup_f64, f64, kernel_f64, embedding_lookup_f64);

#[cfg(feature = "dtype-f16")]
embedding_lookup_fn!(embedding_f16, f16, kernel_f16, embedding_lookup_f16);
#[cfg(feature = "dtype-f32")]
embedding_lookup_fn!(embedding_f32, f32, kernel_f32, embedding_lookup_f32);
#[cfg(feature = "dtype-f64")]
embedding_lookup_fn!(embedding_f64, f64, kernel_f64, embedding_lookup_f64);

#[cfg(feature = "dtype-f16")]
embedding_lookup_fn!(embedding_batch_f16, f16, kernel_f16, embedding_lookup_f16);
#[cfg(feature = "dtype-f32")]
embedding_lookup_fn!(embedding_batch_f32, f32, kernel_f32, embedding_lookup_f32);
#[cfg(feature = "dtype-f64")]
embedding_lookup_fn!(embedding_batch_f64, f64, kernel_f64, embedding_lookup_f64);
