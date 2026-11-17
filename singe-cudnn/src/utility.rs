use std::sync::Arc;

use singe_core::{checked_arithmetic, checked_conversions, report_cleanup_failure};
use singe_cuda::context::Context as CudaContext;
use singe_cudnn_sys as sys;

use crate::{
    backend::descriptor::BackendDescriptor,
    error::{Error, Result},
};

checked_conversions!(Error {
    to_usize -> usize,
    to_i32 -> i32,
    to_u32 -> u32,
    to_i64 -> i64,
    to_size_t -> sys::size_t,
});

checked_arithmetic!(Error {
    checked_byte_len,
    checked_element_count,
});

pub(crate) fn to_i32_vec<T: TryInto<i32> + Copy>(values: &[T], name: &str) -> Result<Vec<i32>> {
    values
        .iter()
        .copied()
        .map(|value| to_i32(value, name))
        .collect()
}

macro_rules! check_range {
    ($name:expr, $expr:expr $(,)?) => {
        if !{ $expr } {
            ::core::result::Result::Err($crate::error::Error::out_of_range(
                ::std::string::String::from($name),
            ))
        } else {
            ::core::result::Result::Ok(())
        }
    };
}

pub(crate) use check_range;

/// Compares known live targets while preserving deviceless and externally managed contexts.
pub(crate) fn validate_context_pair(
    expected: Option<&Arc<CudaContext>>,
    actual: Option<&Arc<CudaContext>>,
    name: &str,
) -> Result<()> {
    match (expected, actual) {
        (Some(expected), Some(actual)) if expected.as_ref() != actual.as_ref() => {
            Err(Error::ContextMismatch {
                name: name.to_owned().into(),
            })
        }
        _ => Ok(()),
    }
}

pub(crate) fn bind_context(context: Option<&Arc<CudaContext>>) -> Result<()> {
    if let Some(context) = context {
        context.bind()?;
    }
    Ok(())
}

pub(crate) fn bind_for_drop(
    context: Option<&Arc<CudaContext>>,
    descriptor: &mut BackendDescriptor,
    object: &str,
) {
    if let Err(error) = bind_context(context) {
        report_cleanup_failure("bind the cuda context to destroy", object, &error);
        descriptor.leak();
    }
}

// Rounds `pointer` up to the next multiple of `alignment`, which must be nonzero.
pub(crate) fn aligned_pointer(pointer: *mut u8, alignment: usize, name: &str) -> Result<*mut u8> {
    let address = pointer.addr();
    let aligned = address
        .checked_next_multiple_of(alignment)
        .ok_or_else(|| Error::out_of_range(name.to_owned()))?;
    Ok(pointer.wrapping_add(aligned - address))
}
