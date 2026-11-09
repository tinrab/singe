use std::sync::{Mutex, MutexGuard, PoisonError};

use singe_core::checked_conversions;

use crate::error::Error;

// Keep typed driver enums and their native conversions/display generated from one list.
macro_rules! driver_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident: $native:ty {
            $($(#[$variant_meta:meta])* $variant:ident = $value:ident),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, num_enum::IntoPrimitive, num_enum::TryFromPrimitive)]
        #[repr(u32)]
        #[non_exhaustive]
        $vis enum $name {
            $($(#[$variant_meta])* $variant = <$native>::$value as _,)*
        }

        impl From<$name> for $native {
            fn from(value: $name) -> Self {
                match value {
                    $($name::$variant => <$native>::$value,)*
                }
            }
        }

        impl TryFrom<$native> for $name {
            type Error = num_enum::TryFromPrimitiveError<Self>;

            fn try_from(value: $native) -> std::result::Result<Self, Self::Error> {
                Self::try_from(value as u32)
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
});

pub(crate) use driver_enum;

/// Converts a native flag constant to the ABI's 16-bit usage field at compile time.
///
/// # Panics
///
/// Panics, including during constant evaluation, if `value` exceeds `u16::MAX`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the preceding assertion checks the ABI field width"
)]
pub const fn usage_bits(value: u32) -> u16 {
    assert!(
        value <= u16::MAX as u32,
        "native usage flag exceeds its ABI field"
    );
    value as u16
}

// Locks `mutex` on a path that must finish even after a panic: cleanup and completion.
// Stopping here could release memory that queued work still uses or leave a waiter blocked forever, which is worse than reading what a panicked holder left.
pub fn lock_for_cleanup<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    #[test]
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
}
