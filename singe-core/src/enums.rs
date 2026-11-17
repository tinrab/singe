use std::{
    error,
    fmt::{self, Display, Formatter},
};

/// A native enum value has no corresponding typed variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnknownEnumValue {
    /// Name of the typed enum whose conversion failed.
    pub name: &'static str,
    /// Unrecognized native value.
    pub value: i64,
}

impl UnknownEnumValue {
    /// Creates an error for `value`, which no variant of the enum `name` represents.
    #[must_use]
    pub fn new(name: &'static str, value: impl Into<i64>) -> Self {
        Self {
            name,
            value: value.into(),
        }
    }
}

impl Display for UnknownEnumValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "unknown `{}` value {}", self.name, self.value)
    }
}

impl error::Error for UnknownEnumValue {}

/// Converts a native integer to its typed variant.
///
/// # Errors
///
/// Returns [`UnknownEnumValue`] when `E::try_from` rejects `value`.
pub fn try_enum_from_raw<E, T>(name: &'static str, value: T) -> Result<E, UnknownEnumValue>
where
    E: TryFrom<T>,
    T: Copy + Into<i64>,
{
    E::try_from(value).map_err(|_| UnknownEnumValue::new(name, value))
}

/// Declares a typed enum over a bindgen newtype enum, with checked conversions in both directions.
///
/// Each variant names the native constant it stands for, as `Variant = CONSTANT`.
/// The macro implements `From<enum> for native` by an exhaustive match, and `TryFrom<native> for enum`, which returns [`UnknownEnumValue`] for a value no variant names.
/// Each variant's discriminant is its native value, so a `#[repr]` on the enum gives it the native integer.
///
/// Write `unknown = Variant` after the native type for an enum whose values may grow with the library, such as a query result.
/// The enum then gains `Variant(code)`, holding any value the other variants do not name, and both conversions are infallible.
#[macro_export]
macro_rules! native_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident: $raw:path, unknown = $unknown:ident($code:ty) {
            $($(#[$variant_meta:meta])* $variant:ident = $constant:ident),* $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis enum $name {
            $($(#[$variant_meta])* $variant,)*
            /// A value this crate does not name, such as one from a newer library.
            $unknown($code),
        }

        impl ::core::convert::From<$name> for $raw {
            fn from(value: $name) -> Self {
                match value {
                    $($name::$variant => <$raw>::$constant,)*
                    $name::$unknown(code) => $raw(code.into()),
                }
            }
        }

        impl ::core::convert::From<$raw> for $name {
            fn from(value: $raw) -> Self {
                $(
                    if value == <$raw>::$constant {
                        return Self::$variant;
                    }
                )*
                Self::$unknown(value.0.into())
            }
        }
    };
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident: $raw:path {
            $($(#[$variant_meta:meta])* $variant:ident = $constant:ident),* $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis enum $name {
            $($(#[$variant_meta])* $variant = <$raw>::$constant.0 as _,)*
        }

        impl ::core::convert::From<$name> for $raw {
            fn from(value: $name) -> Self {
                match value {
                    $($name::$variant => <$raw>::$constant,)*
                }
            }
        }

        impl ::core::convert::TryFrom<$raw> for $name {
            type Error = $crate::UnknownEnumValue;

            fn try_from(value: $raw) -> ::core::result::Result<Self, $crate::UnknownEnumValue> {
                $(
                    if value == <$raw>::$constant {
                        return ::core::result::Result::Ok(Self::$variant);
                    }
                )*
                ::core::result::Result::Err($crate::UnknownEnumValue::new(::core::stringify!($name), value.0))
            }
        }
    };
}

/// Implements [`Display`](std::fmt::Display) for an enum by mapping variants to fixed string names.
#[macro_export]
macro_rules! impl_enum_display {
    ($enum:ty, { $($variant:path => $name:expr),+ $(,)? } $(,)?) => {
        impl ::core::fmt::Display for $enum {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                match self {
                    $($variant => f.write_str($name),)+
                }
            }
        }
    };
}
