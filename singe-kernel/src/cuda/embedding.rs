//! Token embedding table lookups.

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

macro_rules! embedding_lookup_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            token_ids: &impl DeviceSlice<u32>,
            table: &impl DeviceSlice<$ty>,
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
            let stream = borrowed_stream(stream)?;
            cutile::embedding::$name(
                &stream,
                output_pointer(out),
                input_pointer(token_ids),
                input_pointer(table),
                token_ids.len(),
                width,
            )
        }
    };
}

#[cfg(feature = "dtype-f16")]
embedding_lookup_fn!(embedding_lookup_f16, f16);
#[cfg(feature = "dtype-f32")]
embedding_lookup_fn!(embedding_lookup_f32, f32);
#[cfg(feature = "dtype-f64")]
embedding_lookup_fn!(embedding_lookup_f64, f64);

#[cfg(feature = "dtype-f16")]
embedding_lookup_fn!(embedding_f16, f16);
#[cfg(feature = "dtype-f32")]
embedding_lookup_fn!(embedding_f32, f32);
#[cfg(feature = "dtype-f64")]
embedding_lookup_fn!(embedding_f64, f64);

#[cfg(feature = "dtype-f16")]
embedding_lookup_fn!(embedding_batch_f16, f16);
#[cfg(feature = "dtype-f32")]
embedding_lookup_fn!(embedding_batch_f32, f32);
#[cfg(feature = "dtype-f64")]
embedding_lookup_fn!(embedding_batch_f64, f64);
