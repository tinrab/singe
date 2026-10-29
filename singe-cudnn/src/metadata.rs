/// Defines native engine metadata whose numeric IDs can grow with the loaded cuDNN runtime.
macro_rules! backend_metadata_enum {
    (
        $(#[$meta:meta])*
        $visibility:vis enum $name:ident: $native:ty {
            $($(#[$raw_meta:meta])* $variant:ident => $constant:ident => $doc:literal),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
        #[serde(from = "u32", into = "u32")]
        #[non_exhaustive]
        $visibility enum $name {
            $(#[doc = $doc] $variant,)+
            /// A raw native ID, including IDs not named by this crate's headers.
            Unknown(u32),
        }

        impl $name {
            /// Native numeric ID, including IDs unknown to this crate's headers.
            /// Serialization uses this ID, and deserialization selects a named variant when available.
            pub const fn raw(self) -> u32 {
                match self {
                    $($(#[$raw_meta])* Self::$variant => <$native>::$constant as u32,)+
                    Self::Unknown(value) => value,
                }
            }
        }

        impl From<u32> for $name {
            fn from(value: u32) -> Self {
                match value {
                    $(value if value == Self::$variant.raw() => Self::$variant,)+
                    value => Self::Unknown(value),
                }
            }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.raw() == other.raw()
            }
        }

        impl Eq for $name {}

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.raw().cmp(&other.raw())
            }
        }

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                std::hash::Hash::hash(&self.raw(), state);
            }
        }

        impl From<$name> for u32 {
            fn from(value: $name) -> Self {
                value.raw()
            }
        }

        impl From<$native> for $name {
            fn from(value: $native) -> Self {
                Self::from(value as u32)
            }
        }

        impl TryFrom<$name> for $native {
            type Error = $name;

            fn try_from(value: $name) -> Result<Self, Self::Error> {
                Self::try_from(value.raw()).map_err(|_| value)
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match Self::from(self.raw()) {
                    $(Self::$variant => formatter.write_str(stringify!($constant)),)+
                    Self::Unknown(value) => write!(formatter, "{}({value})", stringify!($name)),
                }
            }
        }
    };
}

pub(crate) use backend_metadata_enum;
