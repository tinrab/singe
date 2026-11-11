use std::sync::{Mutex, MutexGuard};

use singe_core::{checked_arithmetic, checked_conversions};

use crate::error::{Error, Result};

checked_conversions!(Error {
    to_i64 -> i64,
    to_u64 -> u64,
    to_usize -> usize,
});

#[cfg(feature = "cuda")]
checked_conversions!(Error {
    to_i32 -> i32,
    to_u32 -> u32,
});

checked_arithmetic!(Error {
    checked_add,
    checked_mul,
    checked_element_count,
});

#[cfg(feature = "cuda")]
pub fn checked_sub(lhs: usize, rhs: usize, name: &str) -> Result<usize> {
    lhs.checked_sub(rhs)
        .ok_or_else(|| Error::out_of_range(name.to_owned()))
}

#[cfg(feature = "cuda")]
pub fn checked_mul_u64(lhs: u64, rhs: u64, name: &str) -> Result<u64> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::out_of_range(name.to_owned()))
}

pub fn checked_add_i64(lhs: i64, rhs: i64, name: &str) -> Result<i64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::out_of_range(name.to_owned()))
}

pub fn checked_sub_i64(lhs: i64, rhs: i64, name: &str) -> Result<i64> {
    lhs.checked_sub(rhs)
        .ok_or_else(|| Error::out_of_range(name.to_owned()))
}

pub fn checked_mul_i64(lhs: i64, rhs: i64, name: &str) -> Result<i64> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::out_of_range(name.to_owned()))
}

#[cfg(any(feature = "cuda", feature = "onnx"))]
pub fn checked_product_i64(dimensions: &[i64], name: &str) -> Result<i64> {
    dimensions.iter().try_fold(1_i64, |product, &dimension| {
        if dimension < 0 {
            return Err(Error::out_of_range(name.to_owned()));
        }
        checked_mul_i64(product, dimension, name)
    })
}

// Locks a mutex, or reports `what` it guards when a panicked holder may have left it inconsistent.
pub trait Locked<T> {
    fn locked(&self, what: &'static str) -> Result<MutexGuard<'_, T>>;
}

impl<T> Locked<T> for Mutex<T> {
    fn locked(&self, what: &'static str) -> Result<MutexGuard<'_, T>> {
        self.lock().map_err(|_| Error::poisoned(what))
    }
}

// Appends a formatted line of generated text to a string.
macro_rules! emit {
    ($out:expr, $($format:tt)*) => {
        {
            use std::fmt::Write as _;
            writeln!($out, $($format)*).expect("formatting into a string cannot fail")
        }
    };
}

pub(crate) use emit;
