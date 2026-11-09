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

#[cfg(test)]
mod tests {
    use std::{sync::Mutex, thread};

    use super::*;

    #[test]
    fn test_a_lock_a_panicked_holder_poisoned_reports_what_it_guards() {
        let cell = Mutex::new(1);
        assert_eq!(*cell.locked("cell").unwrap(), 1);
        let panicked = thread::scope(|scope| {
            scope
                .spawn(|| {
                    let _guard = cell.lock().unwrap();
                    panic!("poison the lock");
                })
                .join()
                .is_err()
        });
        assert!(panicked);
        assert!(matches!(
            cell.locked("cell"),
            Err(Error::Poisoned { what }) if what == "cell"
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_checked_helpers_name_what_overflowed() {
        let out_of_range = |name: &str| Error::out_of_range(name.to_owned()).to_string();
        assert_eq!(
            checked_sub(0, 1, "bytes").unwrap_err().to_string(),
            out_of_range("bytes")
        );
        assert_eq!(
            checked_mul_u64(u64::MAX, 2, "elements")
                .unwrap_err()
                .to_string(),
            out_of_range("elements")
        );
        assert_eq!(
            checked_mul_i64(i64::MAX, 2, "stride")
                .unwrap_err()
                .to_string(),
            out_of_range("stride")
        );
        assert_eq!(checked_sub(3, 1, "bytes").unwrap(), 2);
        assert_eq!(checked_mul_u64(3, 2, "elements").unwrap(), 6);
    }

    #[cfg(any(feature = "cuda", feature = "onnx"))]
    #[test]
    fn test_dimension_products_reject_negative_extents_and_overflow() {
        assert_eq!(checked_product_i64(&[2, 3, 4], "shape").unwrap(), 24);
        assert_eq!(checked_product_i64(&[], "shape").unwrap(), 1);
        assert!(checked_product_i64(&[2, -1], "shape").is_err());
        assert!(checked_product_i64(&[i64::MAX, 2], "shape").is_err());
    }
}
