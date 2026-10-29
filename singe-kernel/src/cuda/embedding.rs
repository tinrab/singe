//! Token embedding table lookups.

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
    utility::{checked_element_count, ensure_len},
};

macro_rules! embedding_lookup_fn {
    ($name:ident, $ty:ty) => {
        /// Copies each token's table row into its corresponding output row.
        ///
        /// # Safety
        ///
        /// Follow the [raw kernel execution contract](crate::cuda#raw-kernel-execution).
        /// Input and output ranges must not overlap.
        /// Token IDs must select valid table rows with signed 32-bit flattened offsets.
        pub unsafe fn $name(
            stream: &Stream,
            out: &mut impl DeviceOutput<$ty>,
            token_ids: &impl DeviceInput<u32>,
            table: &impl DeviceInput<$ty>,
            width: usize,
        ) -> Result<()> {
            if width == 0 {
                return Err(Error::InvalidLength);
            }
            let len = checked_element_count(token_ids.len(), width)?;
            ensure_len(out.len(), len)?;
            if !table.len().is_multiple_of(width) {
                return Err(Error::LengthMismatch);
            }
            let stream = stream.to_cuda_core()?;
            // SAFETY: the caller upholds this function's safety contract.
            unsafe {
                cutile::embedding::$name(
                    &stream,
                    DevicePointer::from_cu_deviceptr(out.as_mut_ptr() as _),
                    input_pointer(token_ids),
                    input_pointer(table),
                    token_ids.len(),
                    width,
                )
            }
        }
    };
}

#[cfg(feature = "dtype_f16")]
embedding_lookup_fn!(embedding_lookup_f16, f16);

#[cfg(feature = "dtype_f32")]
embedding_lookup_fn!(embedding_lookup_f32, f32);

#[cfg(feature = "dtype_f64")]
embedding_lookup_fn!(embedding_lookup_f64, f64);

#[cfg(feature = "dtype_f16")]
embedding_lookup_fn!(embedding_f16, f16);

#[cfg(feature = "dtype_f32")]
embedding_lookup_fn!(embedding_f32, f32);

#[cfg(feature = "dtype_f64")]
embedding_lookup_fn!(embedding_f64, f64);

#[cfg(feature = "dtype_f16")]
embedding_lookup_fn!(embedding_batch_f16, f16);

#[cfg(feature = "dtype_f32")]
embedding_lookup_fn!(embedding_batch_f32, f32);

#[cfg(feature = "dtype_f64")]
embedding_lookup_fn!(embedding_batch_f64, f64);
