/// Declares the attribute enum, its payload-free kind enum, `kind`, and bitwise equality and hashing.
macro_rules! operation_attributes {
    (
        $(#[doc = $doc:literal])*
        pub enum OperationAttributes {
            $(
                $(#[doc = $variant_doc:literal])*
                $variant:ident $(($payload:ty))?
            ),+ $(,)?
        }
    ) => {
        $(#[doc = $doc])*
        #[derive(Debug, Clone)]
        pub enum OperationAttributes {
            $(
                $(#[doc = $variant_doc])*
                $variant $(($payload))?,
            )+
        }

        /// The variant of [`OperationAttributes`] an operation expects, without its payload.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum OperationAttributeKind {
            $(
                $(#[doc = $variant_doc])*
                $variant,
            )+
        }

        impl OperationAttributes {
            /// Returns this value's variant without its payload.
            #[must_use]
            pub fn kind(&self) -> OperationAttributeKind {
                match self {
                    $(Self::$variant { .. } => OperationAttributeKind::$variant,)+
                }
            }
        }

        impl BitHash for OperationAttributes {
            fn bit_hash<H: Hasher>(&self, state: &mut H) {
                self.kind().hash(state);
                match self {
                    $(operation_attributes!(@pattern $variant value $(($payload))?) => {
                        operation_attributes!(@hash state value $(($payload))?);
                    })+
                }
            }

            fn bit_eq(&self, other: &Self) -> bool {
                match (self, other) {
                    $((
                        operation_attributes!(@pattern $variant lhs $(($payload))?),
                        operation_attributes!(@pattern $variant rhs $(($payload))?),
                    ) => operation_attributes!(@eq lhs rhs $(($payload))?),)+
                    _ => false,
                }
            }
        }
    };
    (@pattern $variant:ident $value:ident ($payload:ty)) => { Self::$variant($value) };
    (@pattern $variant:ident $value:ident) => { Self::$variant };
    (@hash $state:ident $value:ident ($payload:ty)) => { $value.bit_hash($state) };
    (@hash $state:ident $value:ident) => { () };
    (@eq $lhs:ident $rhs:ident ($payload:ty)) => { $lhs.bit_eq($rhs) };
    (@eq $lhs:ident $rhs:ident) => { true };
}

/// Declares an attribute struct with bitwise equality and hashing over its fields.
macro_rules! attribute_struct {
    (
        $(#[$meta:meta])*
        pub struct $name:ident {
            $(
                $(#[doc = $field_doc:literal])*
                pub $field:ident: $type:ty
            ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        pub struct $name {
            $(
                $(#[doc = $field_doc])*
                pub $field: $type,
            )+
        }

        impl BitHash for $name {
            fn bit_hash<H: Hasher>(&self, state: &mut H) {
                $(self.$field.bit_hash(state);)+
            }

            fn bit_eq(&self, other: &Self) -> bool {
                $(self.$field.bit_eq(&other.$field))&&+
            }
        }

        bit_hash_traits!($name);
    };
}

/// Implements `PartialEq`, `Eq`, and `Hash` through [`BitHash`].
macro_rules! bit_hash_traits {
    ($($name:ident),+ $(,)?) => {
        $(
            impl PartialEq for $name {
                fn eq(&self, other: &Self) -> bool {
                    self.bit_eq(other)
                }
            }

            impl Eq for $name {}

            impl Hash for $name {
                fn hash<H: Hasher>(&self, state: &mut H) {
                    self.bit_hash(state);
                }
            }
        )+
    };
}

pub(crate) use attribute_struct;
pub(crate) use bit_hash_traits;
pub(crate) use operation_attributes;
