use crate::data_type::DataType;

pub const FLOAT_TYPES: &[DataType] = &[DataType::F16, DataType::Bf16, DataType::F32, DataType::F64];

/// The types kernel IR expressions hold: every type but the packed FP4 encoding.
pub const ELEMENT_TYPES: &[DataType] = &[
    DataType::Bool,
    DataType::U8,
    DataType::I8,
    DataType::U16,
    DataType::I16,
    DataType::U32,
    DataType::I32,
    DataType::U64,
    DataType::I64,
    DataType::F16,
    DataType::Bf16,
    DataType::F32,
    DataType::F64,
    DataType::F8E4M3,
    DataType::F8E5M2,
];

pub const F32_TYPES: &[DataType] = &[DataType::F32];

pub const FP8_TYPES: &[DataType] = &[DataType::F8E4M3, DataType::F8E5M2];

pub const SCALED_MATMUL_OUTPUT_TYPES: &[DataType] = &[DataType::F16, DataType::Bf16, DataType::F32];

pub const I64_TYPES: &[DataType] = &[DataType::I64];
pub const QUANTIZED_I8_U8_TYPES: &[DataType] = &[DataType::I8, DataType::U8];
pub const QUANTIZED_MATMUL_INPUT_TYPES: &[DataType] = QUANTIZED_I8_U8_TYPES;

pub const ALL_DTYPE_TYPES: &[DataType] = &[
    DataType::Bool,
    DataType::U8,
    DataType::I8,
    DataType::U16,
    DataType::I16,
    DataType::U32,
    DataType::I32,
    DataType::U64,
    DataType::I64,
    DataType::F16,
    DataType::Bf16,
    DataType::F32,
    DataType::F64,
    DataType::F8E4M3,
    DataType::F8E5M2,
    DataType::F4E2M1,
];

pub const BOOL_TYPES: &[DataType] = &[DataType::Bool];

pub const CORE_TYPES: &[DataType] = &[DataType::F32, DataType::F64, DataType::I64, DataType::Bool];

pub const F32_F64_TYPES: &[DataType] = &[DataType::F32, DataType::F64];
