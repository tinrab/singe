/// Declares a library status enum from `(Variant = RAW_CONSTANT)` pairs.
///
/// The generated enum has an `Unknown(code)` catch-all and implements `raw()`,
/// `From<code>`, `From<raw enum>`, `TryFrom<Status> for raw enum`, and a
/// [`Display`](std::fmt::Display) that prints the C constant name. Unknown codes
/// print as `$unknown(code)`. Crates add `description()` and predicates in their
/// own `impl Status` block.
#[macro_export]
macro_rules! status_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident: $code:ty => $raw:ty, unknown = $unknown:literal {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident = $constant:ident,
            )*
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum $name {
            $(
                $(#[$variant_meta])*
                $variant,
            )*
            /// A status code not known to this wrapper.
            Unknown($code),
        }

        impl $name {
            /// Returns the raw library status code.
            pub const fn raw(self) -> $code {
                match self {
                    $(Self::$variant => <$raw>::$constant as $code,)*
                    Self::Unknown(code) => code,
                }
            }
        }

        impl From<$code> for $name {
            fn from(code: $code) -> Self {
                $(
                    if code == <$raw>::$constant as $code {
                        return Self::$variant;
                    }
                )*
                Self::Unknown(code)
            }
        }

        impl From<$raw> for $name {
            fn from(status: $raw) -> Self {
                Self::from(status as $code)
            }
        }

        impl TryFrom<$name> for $raw {
            type Error = $name;

            fn try_from(status: $name) -> ::core::result::Result<Self, Self::Error> {
                match status {
                    $($name::$variant => Ok(<$raw>::$constant),)*
                    $name::Unknown(_) => Err(status),
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $(Self::$variant => f.write_str(stringify!($constant)),)*
                    Self::Unknown(code) => write!(f, concat!($unknown, "({})"), code),
                }
            }
        }
    };
}
