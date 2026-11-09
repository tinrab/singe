use std::hash::{Hash, Hasher};

use singe_codegen::interpret::{E4M3, E5M2};

use crate::{
    data_type::{DataType, bf16, f16},
    device::Device,
    error::{Error, Result},
    shape::Shape,
    tensor::{Tensor, TensorDataType, tensor_data_type_cases},
};

/// A scalar argument that retains integer precision and FP64 coefficient bits.
///
/// Scalar arguments are weakly typed during promotion.
/// Explicit rank-zero tensors retain their tensor dtype and participate in tensor promotion.
#[derive(Debug, Clone, Copy)]
pub enum Scalar {
    /// A boolean scalar, retaining its logical value during promotion.
    Bool(bool),
    /// A signed integer with all 64 bits preserved.
    Signed(i64),
    /// An unsigned integer.
    /// I64 arithmetic rejects values outside its range.
    Unsigned(u64),
    /// A floating scalar with its FP64 bits preserved until operand conversion.
    Float(f64),
}

impl Scalar {
    fn key(self) -> (u8, u64) {
        match self {
            Self::Bool(value) => (0, u64::from(value)),
            Self::Signed(value) => (1, value as u64),
            Self::Unsigned(value) => (2, value),
            Self::Float(value) => (3, value.to_bits()),
        }
    }

    /// Returns the data type an operation on a tensor of `input` and this scalar computes in, by `PyTorch`'s rules: a scalar of a higher category than the tensor promotes it to the default type of that category.
    pub(crate) fn promoted_type(self, input: DataType, division: bool) -> Result<DataType> {
        if input == DataType::F4E2M1 {
            return Err(Error::unsupported_data_type("scalar arithmetic input"));
        }

        let result = match self {
            Self::Float(_) if !input.is_float() => DataType::F32,
            Self::Signed(_) | Self::Unsigned(_) if input == DataType::Bool => DataType::I64,
            _ => input,
        };
        if result == DataType::I64
            && matches!(self, Self::Unsigned(value) if value > i64::MAX as u64)
        {
            return Err(Error::out_of_range("integer scalar argument"));
        }
        Ok(if division && !result.is_float() {
            DataType::F32
        } else {
            result
        })
    }

    pub(crate) fn tensor(self, device: Device, data_type: DataType) -> Result<Tensor> {
        self.full(device, data_type, &Shape::scalar())
    }

    pub(crate) fn full(self, device: Device, data_type: DataType, shape: &Shape) -> Result<Tensor> {
        // The 8-bit floating types have no Rust type, so their value is encoded once and repeated.
        let format = match data_type {
            DataType::F8E4M3 => Some(E4M3),
            DataType::F8E5M2 => Some(E5M2),
            _ => None,
        };
        if let Some(format) = format {
            let bits = format.encode(self.float());
            return Tensor::from_bytes(
                device,
                data_type,
                shape.clone(),
                &vec![bits; shape.element_count()],
            );
        }

        macro_rules! create_scalar {
            ($($ty:ty => $variant:ident = $one:expr),+ $(,)?) => {
                match data_type {
                    $(DataType::$variant => Tensor::full(device, shape, <$ty>::from_scalar(self)),)+
                    _ => Err(Error::unsupported_data_type("scalar tensor argument")),
                }
            };
        }
        tensor_data_type_cases!(create_scalar)
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "scalar conversion uses the target IEEE floating-point precision"
    )]
    pub(crate) fn float(self) -> f64 {
        match self {
            Self::Bool(value) => f64::from(u8::from(value)),
            Self::Signed(value) => value as f64,
            Self::Unsigned(value) => value as f64,
            Self::Float(value) => value,
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "scalar conversion uses the target IEEE floating-point precision"
    )]
    pub(crate) fn single(self) -> f32 {
        match self {
            Self::Bool(value) => f32::from(u8::from(value)),
            Self::Signed(value) => value as f32,
            Self::Unsigned(value) => value as f32,
            Self::Float(value) => value as f32,
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "Rust float-to-integer casts saturate and map NaN to zero, matching scalar conversion semantics"
    )]
    pub(crate) fn signed(self) -> i64 {
        match self {
            Self::Bool(value) => i64::from(value),
            Self::Signed(value) => value,
            Self::Unsigned(value) => {
                i64::try_from(value.min(i64::MAX.cast_unsigned())).expect("clamped scalar fits i64")
            }
            Self::Float(value) => value as i64,
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "Rust float-to-integer casts saturate and map NaN to zero, matching scalar conversion semantics"
    )]
    pub(crate) fn unsigned(self) -> u64 {
        match self {
            Self::Bool(value) => u64::from(value),
            Self::Signed(value) => value.max(0) as u64,
            Self::Unsigned(value) => value,
            Self::Float(value) => value as u64,
        }
    }

    pub(crate) fn boolean(self) -> bool {
        match self {
            Self::Bool(value) => value,
            Self::Signed(value) => value != 0,
            Self::Unsigned(value) => value != 0,
            Self::Float(value) => value != 0.0,
        }
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for Scalar {}

impl Hash for Scalar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

impl Tensor {
    pub(crate) fn scalar_operand(&self, scalar: Scalar, division: bool) -> Result<Self> {
        let data_type = scalar.promoted_type(self.data_type(), division)?;
        if data_type == self.data_type() {
            Ok(self.clone())
        } else {
            self.cast(data_type)
        }
    }
}

macro_rules! scalar_from {
    ($variant:ident, $target:ty; $($ty:ty),+) => {
        $(impl From<$ty> for Scalar {
            fn from(value: $ty) -> Self { Self::$variant(value as $target) }
        })+
    };
}

scalar_from!(Signed, i64; i8, i16, i32, i64);

scalar_from!(Unsigned, u64; u8, u16, u32, u64);

scalar_from!(Float, f64; f32, f64);

impl From<bool> for Scalar {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl From<f16> for Scalar {
    fn from(value: f16) -> Self {
        Self::Float(value.to_f64())
    }
}

impl From<bf16> for Scalar {
    fn from(value: bf16) -> Self {
        Self::Float(value.to_f64())
    }
}

macro_rules! scalar_cast {
    (Bool, $value:expr) => {
        $value.boolean()
    };
    (F32, $value:expr) => {
        $value.single()
    };
    (F64, $value:expr) => {
        $value.float()
    };
    (F16, $value:expr) => {
        $crate::data_type::f16::from_f64($value.float())
    };
    (Bf16, $value:expr) => {
        $crate::data_type::bf16::from_f64($value.float())
    };
    (I64, $value:expr) => {
        $value.signed()
    };
    (U64, $value:expr) => {
        $value.unsigned()
    };
    (I8, $value:expr) => {
        i8::try_from(
            $value
                .signed()
                .clamp(i64::from(i8::MIN), i64::from(i8::MAX)),
        )
        .expect("clamped signed scalar fits its target dtype")
    };
    (I16, $value:expr) => {
        i16::try_from(
            $value
                .signed()
                .clamp(i64::from(i16::MIN), i64::from(i16::MAX)),
        )
        .expect("clamped signed scalar fits its target dtype")
    };
    (I32, $value:expr) => {
        i32::try_from(
            $value
                .signed()
                .clamp(i64::from(i32::MIN), i64::from(i32::MAX)),
        )
        .expect("clamped signed scalar fits its target dtype")
    };
    (U8, $value:expr) => {
        u8::try_from($value.unsigned().min(u64::from(u8::MAX)))
            .expect("clamped unsigned scalar fits its target dtype")
    };
    (U16, $value:expr) => {
        u16::try_from($value.unsigned().min(u64::from(u16::MAX)))
            .expect("clamped unsigned scalar fits its target dtype")
    };
    (U32, $value:expr) => {
        u32::try_from($value.unsigned().min(u64::from(u32::MAX)))
            .expect("clamped unsigned scalar fits its target dtype")
    };
}

pub(crate) use scalar_cast;
