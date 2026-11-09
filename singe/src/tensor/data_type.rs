use crate::{
    data_type::DataType,
    error::{Error, Result},
    scalar::{Scalar, scalar_cast},
};

macro_rules! tensor_data_type_cases {
    ($callback:ident) => {
        tensor_data_type_cases! { @cases $callback }
    };
    ($callback:ident, $($argument:tt)+) => {
        tensor_data_type_cases!(@cases $callback $($argument)+;)
    };
    (@cases $callback:ident $($prefix:tt)*) => {
        $callback! {
            $($prefix)*
            bool => Bool = true,
            u8 => U8 = 1_u8,
            i8 => I8 = 1_i8,
            u16 => U16 = 1_u16,
            i16 => I16 = 1_i16,
            u32 => U32 = 1_u32,
            i32 => I32 = 1_i32,
            u64 => U64 = 1_u64,
            i64 => I64 = 1_i64,
            $crate::data_type::f16 => F16 = $crate::data_type::f16::from_f32(1.0),
            $crate::data_type::bf16 => Bf16 = $crate::data_type::bf16::from_f32(1.0),
            f32 => F32 = 1.0_f32,
            f64 => F64 = 1.0_f64,
        }
    };
}

pub(crate) use tensor_data_type_cases;

/// Returns the index and value of the first byte that is not a valid `bool`, which must be 0 or 1.
pub fn invalid_boolean_byte(bytes: &[u8]) -> Option<(usize, u8)> {
    // A branch-free OR over each chunk vectorizes; the position is searched only when a chunk holds an invalid byte.
    if bytes
        .chunks(4096)
        .all(|chunk| chunk.iter().fold(0, |any, &byte| any | byte) <= 1)
    {
        return None;
    }
    bytes
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, byte)| byte > 1)
}

// Every path that imports host bytes as `Bool` values checks them here, since any other byte is not a valid `bool`.
pub fn validate_boolean_bytes(data_type: DataType, bytes: &[u8]) -> Result<()> {
    if data_type == DataType::Bool
        && let Some((index, value)) = invalid_boolean_byte(bytes)
    {
        return Err(Error::invalid_argument(format!(
            "boolean byte {index} (value {value})"
        )));
    }
    Ok(())
}

/// Rust scalar types that can be stored in a tensor.
///
/// This trait is sealed and implemented only for supported primitive scalar types plus the half-precision wrapper types re-exported by [`DataType`].
pub trait TensorDataType: private::Sealed + Copy + Clone + Send + Sync + 'static {
    /// Converts a scalar with saturating integer casts and truncation toward zero.
    fn from_scalar(value: Scalar) -> Self;

    /// Retains this element as a typed scalar value.
    fn into_scalar(self) -> Scalar;

    /// Returns the tensor element data type for this Rust scalar type.
    fn data_type() -> DataType;

    /// Reinterprets a typed slice as bytes.
    fn as_bytes(slice: &[Self]) -> &[u8];

    /// Reinterprets an aligned byte slice as typed elements.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte length is not a multiple of the element size, the byte slice is not aligned, or a boolean byte is neither 0 nor 1.
    fn from_bytes(bytes: &[u8]) -> Result<&[Self]>;

    /// Reinterprets an aligned mutable byte slice as typed elements.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte length is not a multiple of the element size, the byte slice is not aligned, or a boolean byte is neither 0 nor 1.
    fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut [Self]>;
}

mod private {
    pub trait Sealed: Sized {}

    macro_rules! impl_sealed {
        ($($ty:ty => $variant:ident = $one:expr),+ $(,)?) => {
            $(impl Sealed for $ty {})+
        };
    }

    tensor_data_type_cases!(impl_sealed);
}

macro_rules! impl_data_type {
    ($($ty:ty => $variant:ident = $one:expr),+ $(,)?) => {
        $(
            impl TensorDataType for $ty {
                fn from_scalar(value: Scalar) -> Self { scalar_cast!($variant, value) }

                fn into_scalar(self) -> Scalar { Scalar::from(self) }

                fn data_type() -> DataType {
                    DataType::$variant
                }

                fn as_bytes(slice: &[Self]) -> &[u8] {
                    let ptr = slice.as_ptr().cast::<u8>();
                    let len = slice.len() * size_of::<Self>();
                    // SAFETY: any initialized value can be viewed as its bytes.
                    unsafe { std::slice::from_raw_parts(ptr, len) }
                }

                fn from_bytes(bytes: &[u8]) -> Result<&[Self]> {
                    let name = concat!(stringify!($ty), " byte slice");
                    ensure_whole_elements(name, size_of::<Self>(), bytes.len())?;
                    validate_boolean_bytes(DataType::$variant, bytes)?;
                    // SAFETY: every byte was validated as 0 or 1 above, so the aligned body holds valid `bool` values.
                    let (prefix, body, suffix) = unsafe { bytes.align_to::<Self>() };
                    ensure_aligned(name, prefix.len(), suffix.len())?;
                    Ok(body)
                }

                fn from_bytes_mut(bytes: &mut [u8]) -> Result<&mut [Self]> {
                    let name = concat!(stringify!($ty), " mutable byte slice");
                    ensure_whole_elements(name, size_of::<Self>(), bytes.len())?;
                    validate_boolean_bytes(DataType::$variant, bytes)?;
                    // SAFETY: every byte was validated as 0 or 1 above, so the aligned body holds valid `bool` values.
                    let (prefix, body, suffix) = unsafe { bytes.align_to_mut::<Self>() };
                    ensure_aligned(name, prefix.len(), suffix.len())?;
                    Ok(body)
                }
            }
        )+
    };
}

tensor_data_type_cases!(impl_data_type);

fn ensure_whole_elements(name: &'static str, element_size: usize, length: usize) -> Result<()> {
    if length.is_multiple_of(element_size) {
        Ok(())
    } else {
        Err(Error::byte_length_mismatch(name, element_size, length))
    }
}

fn ensure_aligned(name: &'static str, prefix: usize, suffix: usize) -> Result<()> {
    if prefix == 0 && suffix == 0 {
        Ok(())
    } else {
        Err(Error::unaligned_byte_buffer(name, prefix, suffix))
    }
}
