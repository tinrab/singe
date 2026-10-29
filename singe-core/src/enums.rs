use std::{
    error,
    fmt::{self, Display, Formatter},
};

/// A raw enum discriminant has no corresponding typed variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnumConversionError<T> {
    /// Name of the enum whose conversion failed.
    pub name: &'static str,
    /// Unrecognized raw discriminant.
    pub value: T,
}

impl<T: Display> Display for EnumConversionError<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "unknown {} value {}", self.name, self.value)
    }
}

impl<T: Display + fmt::Debug> error::Error for EnumConversionError<T> {}

/// Converts a raw enum discriminant to its typed variant.
///
/// # Errors
///
/// Returns [`EnumConversionError`] when `E::try_from` rejects `value`.
pub fn try_enum_from_raw<E, T>(name: &'static str, value: T) -> Result<E, EnumConversionError<T>>
where
    E: TryFrom<T>,
    T: Copy,
{
    E::try_from(value).map_err(|_| EnumConversionError { name, value })
}

/// Implements bidirectional conversions between a raw FFI enum and a typed Rust enum.
///
/// The typed enum must implement `TryFrom<$ty>` and `Into<$ty>`, where `$ty`
/// matches the raw enum representation. Converting from the raw enum panics if
/// the raw discriminant is not represented by the typed enum. Converting back to
/// the raw enum transmutes the typed discriminant, so the raw enum must use the
/// same integer representation.
#[macro_export]
macro_rules! impl_enum_conversion {
    ($ty:ty, $from:ty, $into:ty $(,)?) => {
        const _: () = {
            impl From<$from> for $into {
                fn from(value: $from) -> Self {
                    let value = value as $ty;
                    <$into>::try_from(value).expect("raw enum value has a typed variant")
                }
            }

            impl From<$into> for $from {
                fn from(value: $into) -> Self {
                    let value = value.into();
                    // SAFETY: `$from` and `$into` share the `$ty` representation, and `value` is the discriminant of a declared variant.
                    unsafe { core::mem::transmute::<$ty, $from>(value) }
                }
            }
        };
    };

    ($from:ty, $into:ty $(,)?) => {
        impl_enum_conversion!(u32, $from, $into);
    };
}

/// Implements [`Display`](std::fmt::Display) for an enum by mapping variants to
/// fixed string names.
#[macro_export]
macro_rules! impl_enum_display {
    ($enum:ty, { $($variant:path => $name:expr),+ $(,)? } $(,)?) => {
        impl std::fmt::Display for $enum {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $($variant => f.write_str($name),)+
                }
            }
        }
    };
}
