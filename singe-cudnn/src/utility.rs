use std::sync::Arc;

use singe_core::checked_conversions;
use singe_cuda::context::Context as CudaContext;

use crate::error::{Error, Result};

checked_conversions!(Error {
    to_usize -> usize,
    to_i32 -> i32,
    to_u32 -> u32,
    to_i64 -> i64,
    to_u64 -> u64,
});

pub fn to_i32_vec<T: TryInto<i32> + Copy>(values: &[T], name: &str) -> Result<Vec<i32>> {
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
pub fn validate_context_pair(
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

pub fn bind_context(context: Option<&Arc<CudaContext>>) -> Result<()> {
    if let Some(context) = context {
        context.bind()?;
    }
    Ok(())
}

pub fn bind_for_drop(context: Option<&Arc<CudaContext>>, resource: &str) {
    if let Err(error) = bind_context(context)
        && cfg!(debug_assertions)
    {
        eprintln!("failed to bind cuda context before destroying {resource}: {error}");
    }
}
