/// Declares crate-local checked integer conversions that report `Error::out_of_range`.
///
/// Each generated function converts with [`TryInto`] and maps failure to `$error::out_of_range(name)`, so every crate reports overflow the same way.
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
/// Each listed helper is generated with a fixed signature, so wrapper crates report size overflow the same way:
///
/// - `checked_add(lhs, rhs, name)` and `checked_mul(lhs, rhs, name)` over `usize`;
/// - `checked_byte_len::<T>(count, name)` for `count * size_of::<T>()`;
/// - `checked_element_count(dimensions, name)` for a product of dimensions of any integer type, each converted to `usize`, so a negative dimension is out of range;
/// - `checked_pitched_extent(rows, pitch, row_len, name)` for the `(rows - 1) * pitch + row_len` elements or bytes that `rows` rows of `row_len` span when consecutive rows start `pitch` apart, which is zero when `rows` or `row_len` is;
/// - `validate_same_len(expected, actual, name)`, which reports a difference as `Error::LengthMismatch { name, expected, actual }`, so it needs that variant;
/// - `validate_min_len(required, actual, name)`, which reports an `actual` length below `required` as `Error::BufferTooSmall { name, required, actual }`, so it needs that variant.
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
        pub(crate) fn checked_element_count<D: ::core::convert::TryInto<usize> + Copy>(
            dimensions: &[D],
            name: &str,
        ) -> ::core::result::Result<usize, $error> {
            dimensions.iter().try_fold(1_usize, |count, &dimension| {
                let dimension: usize = dimension
                    .try_into()
                    .map_err(|_| <$error>::out_of_range(name.to_owned()))?;
                count
                    .checked_mul(dimension)
                    .ok_or_else(|| <$error>::out_of_range(name.to_owned()))
            })
        }
    };
    (@helper $error:ty, checked_pitched_extent) => {
        pub(crate) fn checked_pitched_extent(
            rows: usize,
            pitch: usize,
            row_len: usize,
            name: &str,
        ) -> ::core::result::Result<usize, $error> {
            if rows == 0 || row_len == 0 {
                return ::core::result::Result::Ok(0);
            }
            (rows - 1)
                .checked_mul(pitch)
                .and_then(|offset| offset.checked_add(row_len))
                .ok_or_else(|| <$error>::out_of_range(name.to_owned()))
        }
    };
    (@helper $error:ty, validate_same_len) => {
        pub(crate) fn validate_same_len(
            expected: usize,
            actual: usize,
            name: &str,
        ) -> ::core::result::Result<(), $error> {
            // A struct expression cannot name a variant through a type fragment, but it can through an alias.
            type SameLenError = $error;
            if expected == actual {
                ::core::result::Result::Ok(())
            } else {
                ::core::result::Result::Err(SameLenError::LengthMismatch {
                    name: name.to_owned().into(),
                    expected,
                    actual,
                })
            }
        }
    };
    (@helper $error:ty, validate_min_len) => {
        pub(crate) fn validate_min_len(
            required: usize,
            actual: usize,
            name: &str,
        ) -> ::core::result::Result<(), $error> {
            // A struct expression cannot name a variant through a type fragment, but it can through an alias.
            type MinLenError = $error;
            if actual >= required {
                ::core::result::Result::Ok(())
            } else {
                ::core::result::Result::Err(MinLenError::BufferTooSmall {
                    name: name.to_owned().into(),
                    required,
                    actual,
                })
            }
        }
    };
}
