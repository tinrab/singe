/// Declares crate-local checked integer conversions that report `Error::out_of_range`.
///
/// Each generated function converts with [`TryInto`] and maps failure to
/// `$error::out_of_range(name)`, so every crate reports overflow the same way.
#[macro_export]
macro_rules! checked_conversions {
    ($error:ty { $($name:ident -> $ty:ty),* $(,)? }) => {
        $(
            pub(crate) fn $name(value: impl TryInto<$ty>, name: &str) -> ::core::result::Result<$ty, $error> {
                value.try_into().map_err(|_| <$error>::out_of_range(name.to_owned()))
            }
        )*
    };
}

/// Declares crate-local checked arithmetic helpers that report `Error::out_of_range`.
///
/// Each listed helper is generated with a fixed signature, so wrapper crates report
/// size overflow the same way:
///
/// - `checked_add(lhs, rhs, name)` and `checked_mul(lhs, rhs, name)` over `usize`;
/// - `checked_byte_len::<T>(count, name)` for `count * size_of::<T>()`;
/// - `checked_element_count(dimensions, name)` for a product of non-negative `i64` dimensions.
#[macro_export]
macro_rules! checked_arithmetic {
    ($error:ty { $($helper:ident),* $(,)? }) => {
        $( $crate::checked_arithmetic!(@helper $error, $helper); )*
    };
    (@helper $error:ty, checked_add) => {
        pub(crate) fn checked_add(lhs: usize, rhs: usize, name: &str) -> ::core::result::Result<usize, $error> {
            lhs.checked_add(rhs).ok_or_else(|| <$error>::out_of_range(name.to_owned()))
        }
    };
    (@helper $error:ty, checked_mul) => {
        pub(crate) fn checked_mul(lhs: usize, rhs: usize, name: &str) -> ::core::result::Result<usize, $error> {
            lhs.checked_mul(rhs).ok_or_else(|| <$error>::out_of_range(name.to_owned()))
        }
    };
    (@helper $error:ty, checked_byte_len) => {
        pub(crate) fn checked_byte_len<T>(count: usize, name: &str) -> ::core::result::Result<usize, $error> {
            count
                .checked_mul(::core::mem::size_of::<T>())
                .ok_or_else(|| <$error>::out_of_range(name.to_owned()))
        }
    };
    (@helper $error:ty, checked_element_count) => {
        pub(crate) fn checked_element_count(dimensions: &[i64], name: &str) -> ::core::result::Result<usize, $error> {
            dimensions.iter().try_fold(1_usize, |count, &dimension| {
                let dimension = usize::try_from(dimension)
                    .map_err(|_| <$error>::out_of_range(name.to_owned()))?;
                count
                    .checked_mul(dimension)
                    .ok_or_else(|| <$error>::out_of_range(name.to_owned()))
            })
        }
    };
}
