use std::{
    ffi::c_ulong,
    ops::Range,
    ptr::NonNull,
    sync::{Mutex, MutexGuard, PoisonError},
};

use singe_core::{checked_arithmetic, checked_conversions};

use crate::{
    error::{Error, Result},
    types::DevicePtr,
};

// Keep typed driver enums and their native conversions/display generated from one list.
macro_rules! driver_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident: $native:path {
            $($(#[$variant_meta:meta])* $variant:ident = $value:ident),* $(,)?
        }
    ) => {
        singe_core::native_enum! {
            $(#[$meta])*
            #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
            #[repr(u32)]
            #[non_exhaustive]
            $vis enum $name: $native {
                $($(#[$variant_meta])* $variant = $value,)*
            }
        }

        singe_core::impl_enum_display!($name, { $(Self::$variant => stringify!($value),)* });
    };
}

checked_conversions!(Error {
    to_u32 -> u32,
    to_u64 -> u64,
    to_usize -> usize,
    to_i32 -> i32,
    to_c_ulong -> c_ulong,
});

pub(crate) use driver_enum;

pub(crate) fn checked_bytes<T>(len: usize, name: &str) -> Result<usize> {
    let size = size_of::<T>();
    if size == 0 {
        return if len == 0 {
            Ok(0)
        } else {
            Err(Error::out_of_range(name.to_owned()))
        };
    }
    len.checked_mul(size)
        .ok_or_else(|| Error::out_of_range(name.to_owned()))
}

checked_arithmetic!(Error {
    checked_element_count,
    checked_pitched_extent,
});

/// Converts a native flag constant to the ABI's 16-bit usage field at compile time.
///
/// # Panics
///
/// Panics, including during constant evaluation, if `value` exceeds `u16::MAX`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the preceding assertion checks the ABI field width"
)]
pub(crate) const fn usage_bits(value: u32) -> u16 {
    assert!(
        value <= u16::MAX as u32,
        "native usage flag exceeds its ABI field"
    );
    value as u16
}

pub(crate) fn ranges_overlap(left: &Range<usize>, right: &Range<usize>) -> bool {
    left.start < right.end && right.start < left.end
}

// Checks that `size` bytes at `offset` are nonempty and fit within `total` bytes.
pub(crate) fn check_range(
    offset: usize,
    size: usize,
    total: usize,
    name: &'static str,
) -> Result<()> {
    if size == 0 {
        return Err(Error::not_positive("size"));
    }

    let end = offset
        .checked_add(size)
        .ok_or_else(|| Error::out_of_range("range end"))?;
    if end > total {
        return Err(Error::buffer_too_small(name, end, total));
    }
    Ok(())
}

pub(crate) fn dangling_or_non_null<T>(ptr: *mut T, len: usize) -> Result<NonNull<T>> {
    if len == 0 {
        Ok(NonNull::dangling())
    } else {
        NonNull::new(ptr).ok_or(Error::NullHandle)
    }
}

pub(crate) fn check_address(address: DevicePtr, alignment: usize) -> Result<()> {
    if address.is_null() {
        return Err(Error::null_pointer("memory operation address"));
    }

    if !address.is_aligned_to(alignment) {
        return Err(Error::misaligned("memory operation address", alignment));
    }

    Ok(())
}

// Converts a compiler's NUL-terminated output to text, dropping every trailing NUL.
#[cfg(any(feature = "nvrtc_13_4", feature = "nvvm_13_4"))]
pub(crate) fn bytes_to_string(mut bytes: Vec<u8>) -> String {
    while bytes.last() == Some(&0) {
        bytes.pop();
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

// Locks `mutex` on a path that must finish even after a panic: cleanup and completion.
// Stopping here could release memory that queued work still uses or leave a waiter blocked forever, which is worse than reading what a panicked holder left.
pub(crate) fn lock_for_cleanup<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the guard stays held through the panic so the panic poisons the lock"
    )]
    fn test_cleanup_locks_reach_their_state_after_a_panic() {
        let cell = Mutex::new(1);
        let panicked = thread::scope(|scope| {
            scope
                .spawn(|| {
                    let mut guard = cell.lock().unwrap();
                    *guard = 2;
                    panic!("poison the lock");
                })
                .join()
                .is_err()
        });
        assert!(panicked);
        assert!(cell.lock().is_err());
        assert_eq!(*lock_for_cleanup(&cell), 2);
    }

    #[test]
    fn test_byte_range_conflicts_preserve_boundaries_and_aliases() {
        for (left, right, expected) in [
            (0..4, 4..8, false),
            (0..4, 8..12, false),
            (0..8, 4..12, true),
            (0..16, 4..8, true),
            (4..8, 4..8, true),
            (0..0, 0..0, false),
            // Hazard tracking conservatively includes an empty subrange inside another view.
            (0..8, 4..4, true),
        ] {
            assert_eq!(ranges_overlap(&left, &right), expected);
            assert_eq!(ranges_overlap(&right, &left), expected);
        }
    }
}
