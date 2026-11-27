use singe_nccl_sys::{ncclDataType_t, ncclRedOp_t, ncclResult_t};
use singe_ptx::emulator::{bf16, f16};

/// An element type NCCL reduces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DataType {
    I8,
    U8,
    I32,
    U32,
    I64,
    U64,
    F16,
    F32,
    F64,
    Bf16,
}

impl DataType {
    pub(crate) fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

impl TryFrom<ncclDataType_t> for DataType {
    type Error = ncclResult_t;

    fn try_from(value: ncclDataType_t) -> Result<Self, Self::Error> {
        Ok(match value {
            ncclDataType_t::ncclInt8 => Self::I8,
            ncclDataType_t::ncclUint8 => Self::U8,
            ncclDataType_t::ncclInt32 => Self::I32,
            ncclDataType_t::ncclUint32 => Self::U32,
            ncclDataType_t::ncclInt64 => Self::I64,
            ncclDataType_t::ncclUint64 => Self::U64,
            ncclDataType_t::ncclFloat16 => Self::F16,
            ncclDataType_t::ncclFloat32 => Self::F32,
            ncclDataType_t::ncclFloat64 => Self::F64,
            ncclDataType_t::ncclBfloat16 => Self::Bf16,
            _ => return Err(ncclResult_t::ncclInvalidArgument),
        })
    }
}

/// Where a scaled sum's scalar lives.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Scalar {
    /// The scalar's bytes, copied from the host when the operation was created.
    Host(Vec<u8>),
    /// The scalar's address in device memory, read when the operation runs.
    Device(u64),
}

/// How a reduction combines one element from each rank.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Reduction {
    Sum,
    Product,
    Max,
    Min,
    Average,
    /// Multiplies each rank's input by its scalar, then sums.
    ScaledSum(Scalar),
}

impl Reduction {
    /// Returns whether two ranks asked for the same reduction; scaled sums may scale by different scalars.
    pub(crate) fn matches(&self, other: &Self) -> bool {
        matches!((self, other), (Self::ScaledSum(_), Self::ScaledSum(_))) || self == other
    }
}

impl TryFrom<ncclRedOp_t> for Reduction {
    type Error = ncclResult_t;

    fn try_from(value: ncclRedOp_t) -> Result<Self, Self::Error> {
        Ok(match value {
            ncclRedOp_t::ncclSum => Self::Sum,
            ncclRedOp_t::ncclProd => Self::Product,
            ncclRedOp_t::ncclMax => Self::Max,
            ncclRedOp_t::ncclMin => Self::Min,
            ncclRedOp_t::ncclAvg => Self::Average,
            _ => return Err(ncclResult_t::ncclInvalidArgument),
        })
    }
}

trait Element: Copy {
    fn decode(bytes: &[u8]) -> Self;

    fn encode(self, bytes: &mut [u8]);

    fn add(self, other: Self) -> Self;

    fn multiply(self, other: Self) -> Self;

    fn maximum(self, other: Self) -> Self;

    fn minimum(self, other: Self) -> Self;

    fn divide(self, count: usize) -> Self;
}

macro_rules! integer {
    ($($type:ty),*) => {$(
        impl Element for $type {
            fn decode(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().expect("element width"))
            }
            fn encode(self, bytes: &mut [u8]) {
                bytes.copy_from_slice(&self.to_le_bytes());
            }
            fn add(self, other: Self) -> Self {
                self.wrapping_add(other)
            }
            fn multiply(self, other: Self) -> Self {
                self.wrapping_mul(other)
            }
            fn maximum(self, other: Self) -> Self {
                self.max(other)
            }
            fn minimum(self, other: Self) -> Self {
                self.min(other)
            }
            fn divide(self, count: usize) -> Self {
                // A count beyond the type's range exceeds every value's magnitude, so the quotient is zero.
                Self::try_from(count).map_or(0, |count| self / count)
            }
        }
    )*};
}

integer!(i8, u8, i32, u32, i64, u64);

macro_rules! float {
    ($($type:ty => $wide:ty, $from:expr, $to:expr);* $(;)?) => {$(
        impl Element for $type {
            fn decode(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().expect("element width"))
            }
            fn encode(self, bytes: &mut [u8]) {
                bytes.copy_from_slice(&self.to_le_bytes());
            }
            fn add(self, other: Self) -> Self {
                $to($from(self) + $from(other))
            }
            fn multiply(self, other: Self) -> Self {
                $to($from(self) * $from(other))
            }
            fn maximum(self, other: Self) -> Self {
                $to(<$wide>::max($from(self), $from(other)))
            }
            fn minimum(self, other: Self) -> Self {
                $to(<$wide>::min($from(self), $from(other)))
            }
            fn divide(self, count: usize) -> Self {
                $to($from(self) / count as $wide)
            }
        }
    )*};
}

float! {
    f32 => f32, |value| value, |value| value;
    f64 => f64, |value| value, |value| value;
    f16 => f32, f16::to_f32, f16::from_f32;
    bf16 => f32, bf16::to_f32, bf16::from_f32;
}

fn combine<T: Element>(reduction: &Reduction, inputs: &[(&[u8], &[u8])], output: &mut [u8]) {
    let width = size_of::<T>();
    for (index, chunk) in output.chunks_exact_mut(width).enumerate() {
        let range = index * width..(index + 1) * width;
        let mut values = inputs.iter().map(|(input, scale)| {
            let value = T::decode(&input[range.clone()]);
            match reduction {
                Reduction::ScaledSum(_) => value.multiply(T::decode(scale)),
                _ => value,
            }
        });
        let first = values.next().expect("a collective has at least one rank");
        let result = values.fold(first, |accumulated, value| match reduction {
            Reduction::Sum | Reduction::Average | Reduction::ScaledSum(_) => accumulated.add(value),
            Reduction::Product => accumulated.multiply(value),
            Reduction::Max => accumulated.maximum(value),
            Reduction::Min => accumulated.minimum(value),
        });
        let result = match reduction {
            Reduction::Average => result.divide(inputs.len()),
            _ => result,
        };
        result.encode(chunk);
    }
}

/// Reduces one input per rank into `output`, in rank order; `inputs` pairs each rank's bytes with its scalar for a scaled sum.
pub(crate) fn reduce(
    data_type: DataType,
    reduction: &Reduction,
    inputs: &[(&[u8], &[u8])],
    output: &mut [u8],
) {
    match data_type {
        DataType::I8 => combine::<i8>(reduction, inputs, output),
        DataType::U8 => combine::<u8>(reduction, inputs, output),
        DataType::I32 => combine::<i32>(reduction, inputs, output),
        DataType::U32 => combine::<u32>(reduction, inputs, output),
        DataType::I64 => combine::<i64>(reduction, inputs, output),
        DataType::U64 => combine::<u64>(reduction, inputs, output),
        DataType::F16 => combine::<f16>(reduction, inputs, output),
        DataType::F32 => combine::<f32>(reduction, inputs, output),
        DataType::F64 => combine::<f64>(reduction, inputs, output),
        DataType::Bf16 => combine::<bf16>(reduction, inputs, output),
    }
}
