/// Declares a library status enum from `(Variant = RAW_CONSTANT)` pairs.
///
/// The raw status is a bindgen newtype enum over the integer `code`, so any value the library returns is valid.
/// The generated enum has an `Unknown(code)` catch-all and implements `raw()`, `From<code>`, `From<raw enum>`, `TryFrom<Status> for raw enum`, and a [`Display`](std::fmt::Display) that prints the C constant name.
/// Unknown codes print as `$unknown(code)`.
///
/// Write each variant as `Variant = RAW_CONSTANT => "description"` and add `unknown_description = "..."` after `unknown`, and the macro also generates `description()`.
/// A wrapper crate declares its status with [`wrapper_status!`](crate::wrapper_status), which also generates its error conversions.
#[macro_export]
macro_rules! status_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident: $code:ty => $raw:ty, unknown = $unknown:literal, unknown_description = $unknown_description:tt {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident = $constant:ident => $description:literal,
            )*
        }
    ) => {
        $crate::status_enum! {
            $(#[$meta])*
            pub enum $name: $code => $raw, unknown = $unknown {
                $(
                    $(#[$variant_meta])*
                    $variant = $constant,
                )*
            }
        }

        impl $name {
            /// Returns a description of the status.
            #[must_use]
            pub const fn description(self) -> &'static str {
                match self {
                    $(Self::$variant => $description,)*
                    Self::Unknown(_) => $unknown_description,
                }
            }
        }
    };
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
                    $(Self::$variant => <$raw>::$constant.0 as $code,)*
                    Self::Unknown(code) => code,
                }
            }
        }

        impl ::core::convert::From<$code> for $name {
            fn from(code: $code) -> Self {
                $(
                    if code == <$raw>::$constant.0 as $code {
                        return Self::$variant;
                    }
                )*
                Self::Unknown(code)
            }
        }

        impl ::core::convert::From<$raw> for $name {
            fn from(status: $raw) -> Self {
                Self::from(status.0 as $code)
            }
        }

        impl ::core::convert::TryFrom<$name> for $raw {
            type Error = $name;

            fn try_from(status: $name) -> ::core::result::Result<Self, Self::Error> {
                match status {
                    $($name::$variant => ::core::result::Result::Ok(<$raw>::$constant),)*
                    $name::Unknown(_) => ::core::result::Result::Err(status),
                }
            }
        }

        impl ::core::fmt::Display for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                match self {
                    $(Self::$variant => f.write_str(::core::stringify!($constant)),)*
                    Self::Unknown(code) => ::core::write!(f, ::core::concat!($unknown, "({})"), code),
                }
            }
        }
    };
}

/// Declares a wrapper crate's `Status`, as [`status_enum!`](crate::status_enum) does, and its conversions to the crate's error.
///
/// The invocation names the library, the error variant holding a status, the raw status type, its success constant, and the `-sys` crate's `take_absence`, as in this declaration from `singe-cusparse`, which needs its `-sys` crate to compile:
///
/// ```ignore
/// singe_core::wrapper_status! {
///     library = "cusparse",
///     error = Error::Cusparse,
///     raw = sys::cusparseStatus_t,
///     success = CUSPARSE_STATUS_SUCCESS,
///     absence = sys::take_absence,
///     unknown = "UNKNOWN_CUSPARSE_STATUS",
///     description = sys::cusparseGetErrorString,
///     /// cuSPARSE status code.
///     pub enum Status: u32 {
///         Success = CUSPARSE_STATUS_SUCCESS,
///         NotInitialized = CUSPARSE_STATUS_NOT_INITIALIZED,
///     }
/// }
/// ```
///
/// `description` names the library's error-string function; a library without one leaves it out and writes each variant as `Variant = RAW_CONSTANT => "description"`.
/// The macro implements `From<raw>` for the error, which returns `Error::Load` when the call that produced the status found its library or symbol missing, `From<Status>`, and a crate-private `Error::check(raw) -> Result<(), Error>`.
/// A crate with several statuses names the others' checks with `check = check_<name>` after `absence`.
/// A library that keeps a per-thread diagnostic names the function that reads it with `diagnostic = path` after `absence`, and its variant gains a `diagnostic: String` field.
#[macro_export]
macro_rules! wrapper_status {
    (
        library = $library:literal,
        error = $error:ident :: $variant:ident,
        raw = $raw:ty,
        success = $success:ident,
        absence = $absence:path,
        $(check = $check:ident,)?
        $(diagnostic = $diagnostic:path,)?
        unknown = $unknown:literal,
        description = $describe:path,
        $(#[$meta:meta])*
        pub enum $status:ident: $code:ty {
            $(
                $(#[$variant_meta:meta])*
                $status_variant:ident = $constant:ident,
            )*
        }
    ) => {
        $crate::status_enum! {
            $(#[$meta])*
            pub enum $status: $code => $raw, unknown = $unknown {
                $(
                    $(#[$variant_meta])*
                    $status_variant = $constant,
                )*
            }
        }

        impl $status {
            /// Returns the library's description of this status.
            #[must_use]
            pub fn description(self) -> &'static str {
                const UNKNOWN: &str = ::core::concat!("unknown ", $library, " error");
                let ::core::result::Result::Ok(raw) = <$raw as ::core::convert::TryFrom<$status>>::try_from(self) else {
                    return UNKNOWN;
                };
                // SAFETY: the library's error-string function returns null or a static NUL-terminated string.
                unsafe { $crate::static_str_from_c_ptr($describe(raw), UNKNOWN) }
            }
        }

        $crate::wrapper_status!(@conversions $error, $variant, $raw, $success, $absence, [$($check)?], [$($diagnostic)?], $status);
    };
    (
        library = $library:literal,
        error = $error:ident :: $variant:ident,
        raw = $raw:ty,
        success = $success:ident,
        absence = $absence:path,
        $(check = $check:ident,)?
        $(diagnostic = $diagnostic:path,)?
        unknown = $unknown:literal,
        $(#[$meta:meta])*
        pub enum $status:ident: $code:ty {
            $(
                $(#[$variant_meta:meta])*
                $status_variant:ident = $constant:ident => $description:literal,
            )*
        }
    ) => {
        $crate::status_enum! {
            $(#[$meta])*
            pub enum $status: $code => $raw, unknown = $unknown, unknown_description = (::core::concat!("unknown ", $library, " error")) {
                $(
                    $(#[$variant_meta])*
                    $status_variant = $constant => $description,
                )*
            }
        }

        $crate::wrapper_status!(@conversions $error, $variant, $raw, $success, $absence, [$($check)?], [$($diagnostic)?], $status);
    };
    (@conversions $error:ident, $variant:ident, $raw:ty, $success:ident, $absence:path, [$($check:ident)?], [$($diagnostic:path)?], $status:ident) => {
        impl ::core::convert::From<$raw> for $error {
            fn from(status: $raw) -> Self {
                if let ::core::option::Option::Some(absence) = $absence() {
                    return Self::Load(absence.into_error());
                }

                $crate::wrapper_status!(@native $error, $variant, $status::from(status), [$($diagnostic)?])
            }
        }

        impl ::core::convert::From<$status> for $error {
            fn from(code: $status) -> Self {
                $crate::wrapper_status!(@typed $error, $variant, code, [$($diagnostic)?])
            }
        }

        $crate::wrapper_status!(@check $error, $raw, $success, [$($check)?]);
    };
    (@native $error:ident, $variant:ident, $code:expr, []) => {
        $error::$variant { code: $code }
    };
    (@native $error:ident, $variant:ident, $code:expr, [$diagnostic:path]) => {
        $error::$variant { code: $code, diagnostic: $diagnostic() }
    };
    (@typed $error:ident, $variant:ident, $code:expr, []) => {
        $error::$variant { code: $code }
    };
    (@typed $error:ident, $variant:ident, $code:expr, [$diagnostic:path]) => {
        $error::$variant { code: $code, diagnostic: ::std::string::String::new() }
    };
    (@check $error:ident, $raw:ty, $success:ident, []) => {
        $crate::wrapper_status!(@check $error, $raw, $success, [check]);
    };
    (@check $error:ident, $raw:ty, $success:ident, [$check:ident]) => {
        impl $error {
            pub(crate) fn $check(status: $raw) -> ::core::result::Result<(), Self> {
                if status == <$raw>::$success {
                    ::core::result::Result::Ok(())
                } else {
                    ::core::result::Result::Err(Self::from(status))
                }
            }
        }
    };
}
