use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cublas_sys as sys;

/// Matrix data ordering.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum Order {
    ColumnMajor = sys::cublasLtOrder_t::CUBLASLT_ORDER_COL as _,
    RowMajor = sys::cublasLtOrder_t::CUBLASLT_ORDER_ROW as _,
    Column32 = sys::cublasLtOrder_t::CUBLASLT_ORDER_COL32 as _,
    Column4_4R2_8C = sys::cublasLtOrder_t::CUBLASLT_ORDER_COL4_4R2_8C as _,
    Column32TwoRowsFourRowsFourColumns = sys::cublasLtOrder_t::CUBLASLT_ORDER_COL32_2R_4R4 as _,
}

impl_enum_conversion!(sys::cublasLtOrder_t, Order);

impl_enum_display!(Order, {
    Order::ColumnMajor => "CUBLASLT_ORDER_COL",
    Order::RowMajor => "CUBLASLT_ORDER_ROW",
    Order::Column32 => "CUBLASLT_ORDER_COL32",
    Order::Column4_4R2_8C => "CUBLASLT_ORDER_COL4_4R2_8C",
    Order::Column32TwoRowsFourRowsFourColumns => "CUBLASLT_ORDER_COL32_2R_4R4",
});

/// Batch processing mode for matrix layouts.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum BatchMode {
    Strided = sys::cublasLtBatchMode_t::CUBLASLT_BATCH_MODE_STRIDED as _,
    PointerArray = sys::cublasLtBatchMode_t::CUBLASLT_BATCH_MODE_POINTER_ARRAY as _,
    Grouped = sys::cublasLtBatchMode_t::CUBLASLT_BATCH_MODE_GROUPED as _,
}

impl_enum_conversion!(sys::cublasLtBatchMode_t, BatchMode);

impl_enum_display!(BatchMode, {
    BatchMode::Strided => "CUBLASLT_BATCH_MODE_STRIDED",
    BatchMode::PointerArray => "CUBLASLT_BATCH_MODE_POINTER_ARRAY",
    BatchMode::Grouped => "CUBLASLT_BATCH_MODE_GROUPED",
});

/// Experimental integer width for grouped-matrix dimension arrays.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum IntegerWidth {
    Bits32 = sys::cublasLtIntegerWidth_t::CUBLASLT_INTEGER_WIDTH_32 as _,
    Bits64 = sys::cublasLtIntegerWidth_t::CUBLASLT_INTEGER_WIDTH_64 as _,
}

impl_enum_conversion!(sys::cublasLtIntegerWidth_t, IntegerWidth);

impl_enum_display!(IntegerWidth, {
    IntegerWidth::Bits32 => "CUBLASLT_INTEGER_WIDTH_32",
    IntegerWidth::Bits64 => "CUBLASLT_INTEGER_WIDTH_64",
});

/// [`MatrixLayoutAttribute`] identifies the configurable attributes of a
/// [`MatrixLayout`](crate::lt::descriptor::MatrixLayout).
/// Use [`MatrixLayout::attribute`](crate::lt::descriptor::MatrixLayout::attribute) and
/// [`MatrixLayout::set_attribute`](crate::lt::descriptor::MatrixLayout::set_attribute) to query
/// and update them.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum MatrixLayoutAttribute {
    Type = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_TYPE as _,
    Order = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER as _,
    Rows = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ROWS as _,
    Cols = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_COLS as _,
    LeadingDimension = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_LD as _,
    BatchCount = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT as _,
    StridedBatchOffset =
        sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET as _,
    PlaneOffset = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET as _,
    BatchMode = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_MODE as _,
    GroupedRowsArray =
        sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_ARRAY as _,
    GroupedColsArray =
        sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_GROUPED_MATRIX_LAYOUT_COLS_ARRAY as _,
    GroupedLeadingDimensionArray =
        sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY as _,
    GroupedRowsColsArrayIntegerWidth = sys::cublasLtMatrixLayoutAttribute_t::
        CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH as _,
    GroupedLeadingDimensionArrayIntegerWidth = sys::cublasLtMatrixLayoutAttribute_t::
        CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH as _,
}

impl_enum_conversion!(sys::cublasLtMatrixLayoutAttribute_t, MatrixLayoutAttribute);

impl_enum_display!(MatrixLayoutAttribute, {
    MatrixLayoutAttribute::Type => "CUBLASLT_MATRIX_LAYOUT_TYPE",
    MatrixLayoutAttribute::Order => "CUBLASLT_MATRIX_LAYOUT_ORDER",
    MatrixLayoutAttribute::Rows => "CUBLASLT_MATRIX_LAYOUT_ROWS",
    MatrixLayoutAttribute::Cols => "CUBLASLT_MATRIX_LAYOUT_COLS",
    MatrixLayoutAttribute::LeadingDimension => "CUBLASLT_MATRIX_LAYOUT_LD",
    MatrixLayoutAttribute::BatchCount => "CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT",
    MatrixLayoutAttribute::StridedBatchOffset => "CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET",
    MatrixLayoutAttribute::PlaneOffset => "CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET",
    MatrixLayoutAttribute::BatchMode => "CUBLASLT_MATRIX_LAYOUT_BATCH_MODE",
    MatrixLayoutAttribute::GroupedRowsArray => "CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_ARRAY",
    MatrixLayoutAttribute::GroupedColsArray => "CUBLASLT_GROUPED_MATRIX_LAYOUT_COLS_ARRAY",
    MatrixLayoutAttribute::GroupedLeadingDimensionArray => "CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY",
    MatrixLayoutAttribute::GroupedRowsColsArrayIntegerWidth => "CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH",
    MatrixLayoutAttribute::GroupedLeadingDimensionArrayIntegerWidth => "CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH",
});

/// Pointer mode for the scaling factors `alpha` and `beta`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum PointerMode {
    Host = sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST as _,
    Device = sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE as _,
    DeviceVector = sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE_VECTOR as _,
    AlphaDeviceVectorBetaZero =
        sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO as _,
    AlphaDeviceVectorBetaHost =
        sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST as _,
}

impl_enum_conversion!(sys::cublasLtPointerMode_t, PointerMode);

impl_enum_display!(PointerMode, {
    PointerMode::Host => "CUBLASLT_POINTER_MODE_HOST",
    PointerMode::Device => "CUBLASLT_POINTER_MODE_DEVICE",
    PointerMode::DeviceVector => "CUBLASLT_POINTER_MODE_DEVICE_VECTOR",
    PointerMode::AlphaDeviceVectorBetaZero => "CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO",
    PointerMode::AlphaDeviceVectorBetaHost => "CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST",
});

/// [`MatmulDescriptorAttribute`] identifies the configurable attributes of a
/// [`MatmulDescriptor`](crate::lt::matmul::MatmulDescriptor).
/// Use [`MatmulDescriptor::attribute`](crate::lt::matmul::MatmulDescriptor::attribute)
/// and [`MatmulDescriptor::set_attribute`](crate::lt::matmul::MatmulDescriptor::set_attribute) to
/// query and update them.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum MatmulDescriptorAttribute {
    ComputeType = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_COMPUTE_TYPE as _,
    ScaleType = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_SCALE_TYPE as _,
    PointerMode = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_POINTER_MODE as _,
    TransposeA = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA as _,
    TransposeB = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB as _,
    TransposeC = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSC as _,
    FillMode = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FILL_MODE as _,
    Epilogue = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE as _,
    BiasPointer = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER as _,
    BiasBatchStride =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE as _,
    EpilogueAuxPointer =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER as _,
    EpilogueAuxLeadingDimension =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD as _,
    EpilogueAuxBatchStride =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE as _,
    AlphaVectorBatchStride =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE as _,
    SmCountTarget = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET as _,
    AScalePointer = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER as _,
    BScalePointer = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER as _,
    CScalePointer = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_C_SCALE_POINTER as _,
    DScalePointer = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_D_SCALE_POINTER as _,
    AmaxDPointer = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_AMAX_D_POINTER as _,
    EpilogueAuxDataType =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE as _,
    EpilogueAuxScalePointer =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER as _,
    EpilogueAuxAmaxPointer =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER as _,
    FastAccum = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM as _,
    BiasDataType = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE as _,
    AScaleMode = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE as _,
    BScaleMode = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE as _,
    CScaleMode = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_C_SCALE_MODE as _,
    DScaleMode = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_D_SCALE_MODE as _,
    EpilogueAuxScaleMode =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE as _,
    DOutScalePointer =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER as _,
    DOutScaleMode = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE as _,
    EmulationDescriptor =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR as _,
    AlphaBatchStride =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE as _,
    BetaBatchStride =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE as _,
}

impl_enum_conversion!(
    sys::cublasLtMatmulDescAttributes_t,
    MatmulDescriptorAttribute,
);

impl_enum_display!(MatmulDescriptorAttribute, {
    MatmulDescriptorAttribute::ComputeType => "CUBLASLT_MATMUL_DESC_COMPUTE_TYPE",
    MatmulDescriptorAttribute::ScaleType => "CUBLASLT_MATMUL_DESC_SCALE_TYPE",
    MatmulDescriptorAttribute::PointerMode => "CUBLASLT_MATMUL_DESC_POINTER_MODE",
    MatmulDescriptorAttribute::TransposeA => "CUBLASLT_MATMUL_DESC_TRANSA",
    MatmulDescriptorAttribute::TransposeB => "CUBLASLT_MATMUL_DESC_TRANSB",
    MatmulDescriptorAttribute::TransposeC => "CUBLASLT_MATMUL_DESC_TRANSC",
    MatmulDescriptorAttribute::FillMode => "CUBLASLT_MATMUL_DESC_FILL_MODE",
    MatmulDescriptorAttribute::Epilogue => "CUBLASLT_MATMUL_DESC_EPILOGUE",
    MatmulDescriptorAttribute::BiasPointer => "CUBLASLT_MATMUL_DESC_BIAS_POINTER",
    MatmulDescriptorAttribute::BiasBatchStride => "CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE",
    MatmulDescriptorAttribute::EpilogueAuxPointer => "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER",
    MatmulDescriptorAttribute::EpilogueAuxLeadingDimension => "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD",
    MatmulDescriptorAttribute::EpilogueAuxBatchStride => "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE",
    MatmulDescriptorAttribute::AlphaVectorBatchStride => "CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE",
    MatmulDescriptorAttribute::SmCountTarget => "CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET",
    MatmulDescriptorAttribute::AScalePointer => "CUBLASLT_MATMUL_DESC_A_SCALE_POINTER",
    MatmulDescriptorAttribute::BScalePointer => "CUBLASLT_MATMUL_DESC_B_SCALE_POINTER",
    MatmulDescriptorAttribute::CScalePointer => "CUBLASLT_MATMUL_DESC_C_SCALE_POINTER",
    MatmulDescriptorAttribute::DScalePointer => "CUBLASLT_MATMUL_DESC_D_SCALE_POINTER",
    MatmulDescriptorAttribute::AmaxDPointer => "CUBLASLT_MATMUL_DESC_AMAX_D_POINTER",
    MatmulDescriptorAttribute::EpilogueAuxDataType => "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE",
    MatmulDescriptorAttribute::EpilogueAuxScalePointer => "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER",
    MatmulDescriptorAttribute::EpilogueAuxAmaxPointer => "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER",
    MatmulDescriptorAttribute::FastAccum => "CUBLASLT_MATMUL_DESC_FAST_ACCUM",
    MatmulDescriptorAttribute::BiasDataType => "CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE",
    MatmulDescriptorAttribute::AScaleMode => "CUBLASLT_MATMUL_DESC_A_SCALE_MODE",
    MatmulDescriptorAttribute::BScaleMode => "CUBLASLT_MATMUL_DESC_B_SCALE_MODE",
    MatmulDescriptorAttribute::CScaleMode => "CUBLASLT_MATMUL_DESC_C_SCALE_MODE",
    MatmulDescriptorAttribute::DScaleMode => "CUBLASLT_MATMUL_DESC_D_SCALE_MODE",
    MatmulDescriptorAttribute::EpilogueAuxScaleMode => "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE",
    MatmulDescriptorAttribute::DOutScalePointer => "CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER",
    MatmulDescriptorAttribute::DOutScaleMode => "CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE",
    MatmulDescriptorAttribute::EmulationDescriptor => "CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR",
    MatmulDescriptorAttribute::AlphaBatchStride => "CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE",
    MatmulDescriptorAttribute::BetaBatchStride => "CUBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE",
});

/// Scaling mode that defines how scaling factor pointers are interpreted.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum MatrixScale {
    Scalar32F = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F as _,
    OuterVector32F =
        sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F as _,
    Vector32UE8M0 = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 as _,
    Vector16UE4M3 = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 as _,
    Vector12832F = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F as _,
    Block128x12832F =
        sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F as _,
    PerBatchScalar32F =
        sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_PER_BATCH_SCALAR_32F as _,
}

impl_enum_conversion!(sys::cublasLtMatmulMatrixScale_t, MatrixScale);

impl_enum_display!(MatrixScale, {
    MatrixScale::Scalar32F => "CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F",
    MatrixScale::OuterVector32F => "CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F",
    MatrixScale::Vector32UE8M0 => "CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0",
    MatrixScale::Vector16UE4M3 => "CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3",
    MatrixScale::Vector12832F => "CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F",
    MatrixScale::Block128x12832F => "CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F",
    MatrixScale::PerBatchScalar32F => "CUBLASLT_MATMUL_MATRIX_SCALE_PER_BATCH_SCALAR_32F",
});

/// [`MatrixTransformDescAttribute`] identifies the configurable attributes of a
/// [`MatrixTransformDescriptor`](crate::lt::descriptor::MatrixTransformDescriptor).
/// Use
/// [`MatrixTransformDescriptor::attribute`](crate::lt::descriptor::MatrixTransformDescriptor::attribute)
/// and
/// [`MatrixTransformDescriptor::set_attribute`](crate::lt::descriptor::MatrixTransformDescriptor::set_attribute)
/// to query and update them.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum MatrixTransformDescAttribute {
    ScaleType =
        sys::cublasLtMatrixTransformDescAttributes_t::CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE
            as _,
    PointerMode =
        sys::cublasLtMatrixTransformDescAttributes_t::CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE
            as _,
    TransposeA =
        sys::cublasLtMatrixTransformDescAttributes_t::CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA as _,
    TransposeB =
        sys::cublasLtMatrixTransformDescAttributes_t::CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB as _,
}

impl_enum_conversion!(
    sys::cublasLtMatrixTransformDescAttributes_t,
    MatrixTransformDescAttribute
);

impl_enum_display!(MatrixTransformDescAttribute, {
    MatrixTransformDescAttribute::ScaleType => "CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE",
    MatrixTransformDescAttribute::PointerMode => "CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE",
    MatrixTransformDescAttribute::TransposeA => "CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA",
    MatrixTransformDescAttribute::TransposeB => "CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB",
});

/// [`EmulationDescAttribute`] identifies the configurable floating-point emulation parameters of
/// an emulation descriptor.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum EmulationDescAttribute {
    Strategy = sys::cublasLtEmulationDescAttributes_t::CUBLASLT_EMULATION_DESC_STRATEGY as _,
    SpecialValuesSupport =
        sys::cublasLtEmulationDescAttributes_t::CUBLASLT_EMULATION_DESC_SPECIAL_VALUES_SUPPORT
            as _,
    FixedPointMantissaControl = sys::cublasLtEmulationDescAttributes_t::
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL as _,
    FixedPointMaxMantissaBitCount = sys::cublasLtEmulationDescAttributes_t::
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT as _,
    FixedPointMantissaBitOffset = sys::cublasLtEmulationDescAttributes_t::
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_OFFSET as _,
    FixedPointMantissaBitCountPointer = sys::cublasLtEmulationDescAttributes_t::
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_COUNT_POINTER as _,
}

impl_enum_conversion!(
    sys::cublasLtEmulationDescAttributes_t,
    EmulationDescAttribute
);

impl_enum_display!(EmulationDescAttribute, {
    EmulationDescAttribute::Strategy => "CUBLASLT_EMULATION_DESC_STRATEGY",
    EmulationDescAttribute::SpecialValuesSupport => "CUBLASLT_EMULATION_DESC_SPECIAL_VALUES_SUPPORT",
    EmulationDescAttribute::FixedPointMantissaControl => "CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL",
    EmulationDescAttribute::FixedPointMaxMantissaBitCount => "CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT",
    EmulationDescAttribute::FixedPointMantissaBitOffset => "CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_OFFSET",
    EmulationDescAttribute::FixedPointMantissaBitCountPointer => "CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_COUNT_POINTER",
});

/// [`ReductionScheme`] specifies a reduction scheme for the portions of the dot product calculated in parallel, also known as split-K.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum ReductionScheme {
    None = sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_NONE as _,
    InPlace = sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_INPLACE as _,
    ComputeType = sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE as _,
    OutputType = sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE as _,
    Mask = sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_MASK as _,
}

impl_enum_conversion!(sys::cublasLtReductionScheme_t, ReductionScheme);

impl_enum_display!(ReductionScheme, {
    ReductionScheme::None => "CUBLASLT_REDUCTION_SCHEME_NONE",
    ReductionScheme::InPlace => "CUBLASLT_REDUCTION_SCHEME_INPLACE",
    ReductionScheme::ComputeType => "CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE",
    ReductionScheme::OutputType => "CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE",
    ReductionScheme::Mask => "CUBLASLT_REDUCTION_SCHEME_MASK",
});

/// [`Epilogue`] selects post-processing applied after the main matrix multiplication.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum Epilogue {
    Default = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT as _,
    Relu = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU as _,
    ReluAux = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_AUX as _,
    Bias = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS as _,
    ReluBias = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS as _,
    ReluAuxBias = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_AUX_BIAS as _,
    DRelu = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DRELU as _,
    DReluBGrad = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DRELU_BGRAD as _,
    Gelu = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU as _,
    GeluAux = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_AUX as _,
    GeluBias = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS as _,
    GeluAuxBias = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_AUX_BIAS as _,
    DGelu = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DGELU as _,
    DGeluBGrad = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DGELU_BGRAD as _,
    BGradA = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BGRADA as _,
    BGradB = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BGRADB as _,
}

impl_enum_conversion!(sys::cublasLtEpilogue_t, Epilogue);

impl_enum_display!(Epilogue, {
    Epilogue::Default => "CUBLASLT_EPILOGUE_DEFAULT",
    Epilogue::Relu => "CUBLASLT_EPILOGUE_RELU",
    Epilogue::ReluAux => "CUBLASLT_EPILOGUE_RELU_AUX",
    Epilogue::Bias => "CUBLASLT_EPILOGUE_BIAS",
    Epilogue::ReluBias => "CUBLASLT_EPILOGUE_RELU_BIAS",
    Epilogue::ReluAuxBias => "CUBLASLT_EPILOGUE_RELU_AUX_BIAS",
    Epilogue::DRelu => "CUBLASLT_EPILOGUE_DRELU",
    Epilogue::DReluBGrad => "CUBLASLT_EPILOGUE_DRELU_BGRAD",
    Epilogue::Gelu => "CUBLASLT_EPILOGUE_GELU",
    Epilogue::GeluAux => "CUBLASLT_EPILOGUE_GELU_AUX",
    Epilogue::GeluBias => "CUBLASLT_EPILOGUE_GELU_BIAS",
    Epilogue::GeluAuxBias => "CUBLASLT_EPILOGUE_GELU_AUX_BIAS",
    Epilogue::DGelu => "CUBLASLT_EPILOGUE_DGELU",
    Epilogue::DGeluBGrad => "CUBLASLT_EPILOGUE_DGELU_BGRAD",
    Epilogue::BGradA => "CUBLASLT_EPILOGUE_BGRADA",
    Epilogue::BGradB => "CUBLASLT_EPILOGUE_BGRADB",
});

/// Heuristics search mode.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum SearchMode {
    BestFit = sys::cublasLtMatmulSearch_t::CUBLASLT_SEARCH_BEST_FIT as _,
    LimitedByAlgorithmId = sys::cublasLtMatmulSearch_t::CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID as _,
}

impl_enum_conversion!(sys::cublasLtMatmulSearch_t, SearchMode);

impl_enum_display!(SearchMode, {
    SearchMode::BestFit => "CUBLASLT_SEARCH_BEST_FIT",
    SearchMode::LimitedByAlgorithmId => "CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID",
});

/// [`MatmulPreferenceAttribute`] identifies algorithm-search preferences stored in a
/// [`MatmulPreference`](crate::lt::matmul::MatmulPreference).
/// Use [`MatmulPreference::attribute`](crate::lt::matmul::MatmulPreference::attribute)
/// and [`MatmulPreference::set_attribute`](crate::lt::matmul::MatmulPreference::set_attribute) to
/// query and update them.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum MatmulPreferenceAttribute {
    SearchMode = sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_SEARCH_MODE as _,
    MaxWorkspaceBytes =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES as _,
    ReductionSchemeMask =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK as _,
    MinAlignmentABytes =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES as _,
    MinAlignmentBBytes =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES as _,
    MinAlignmentCBytes =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES as _,
    MinAlignmentDBytes =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES as _,
    MaxWavesCount =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT as _,
    ImplementationMask =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_IMPL_MASK as _,
    GroupedAverageReductionDim = sys::cublasLtMatmulPreferenceAttributes_t::
        CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM as _,
    GroupedDescDAverageRows = sys::cublasLtMatmulPreferenceAttributes_t::
        CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS as _,
    GroupedDescDAverageCols = sys::cublasLtMatmulPreferenceAttributes_t::
        CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS as _,
}

impl_enum_conversion!(
    sys::cublasLtMatmulPreferenceAttributes_t,
    MatmulPreferenceAttribute
);

impl_enum_display!(MatmulPreferenceAttribute, {
    MatmulPreferenceAttribute::SearchMode => "CUBLASLT_MATMUL_PREF_SEARCH_MODE",
    MatmulPreferenceAttribute::MaxWorkspaceBytes => "CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES",
    MatmulPreferenceAttribute::ReductionSchemeMask => "CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK",
    MatmulPreferenceAttribute::MinAlignmentABytes => "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES",
    MatmulPreferenceAttribute::MinAlignmentBBytes => "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES",
    MatmulPreferenceAttribute::MinAlignmentCBytes => "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES",
    MatmulPreferenceAttribute::MinAlignmentDBytes => "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES",
    MatmulPreferenceAttribute::MaxWavesCount => "CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT",
    MatmulPreferenceAttribute::ImplementationMask => "CUBLASLT_MATMUL_PREF_IMPL_MASK",
    MatmulPreferenceAttribute::GroupedAverageReductionDim => "CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM",
    MatmulPreferenceAttribute::GroupedDescDAverageRows => "CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS",
    MatmulPreferenceAttribute::GroupedDescDAverageCols => "CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS",
});

/// [`MatmulAlgorithmCapAttribute`] enumerates capability attributes available on an
/// initialized [`MatmulAlgorithm`](crate::lt::matmul::MatmulAlgorithm) with
/// [`MatmulAlgorithm::cap_attribute`](crate::lt::matmul::MatmulAlgorithm::cap_attribute).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum MatmulAlgorithmCapAttribute {
    SplitKSupport = sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_SPLITK_SUPPORT as _,
    ReductionSchemeMask =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK as _,
    CtaSwizzlingSupport =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT as _,
    StridedBatchSupport =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT as _,
    OutOfPlaceResultSupport =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT as _,
    UploSupport = sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_UPLO_SUPPORT as _,
    TileIds = sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_TILE_IDS as _,
    CustomOptionMax =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX as _,
    CustomMemoryOrder =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER as _,
    PointerModeMask =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_POINTER_MODE_MASK as _,
    EpilogueMask = sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_EPILOGUE_MASK as _,
    StagesIds = sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_STAGES_IDS as _,
    LeadingDimensionNegative =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_LD_NEGATIVE as _,
    NumericalImplementationFlags =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS as _,
    MinAlignmentABytes =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES as _,
    MinAlignmentBBytes =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES as _,
    MinAlignmentCBytes =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES as _,
    MinAlignmentDBytes =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES as _,
    PointerArrayBatchSupport =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_POINTER_ARRAY_BATCH_SUPPORT as _,
    FloatingPointEmulationSupport =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_FLOATING_POINT_EMULATION_SUPPORT
            as _,
    PointerArrayGroupedSupport =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_POINTER_ARRAY_GROUPED_SUPPORT
            as _,
}

impl_enum_conversion!(
    sys::cublasLtMatmulAlgoCapAttributes_t,
    MatmulAlgorithmCapAttribute
);

impl_enum_display!(MatmulAlgorithmCapAttribute, {
    MatmulAlgorithmCapAttribute::SplitKSupport => "CUBLASLT_ALGO_CAP_SPLITK_SUPPORT",
    MatmulAlgorithmCapAttribute::ReductionSchemeMask => "CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK",
    MatmulAlgorithmCapAttribute::CtaSwizzlingSupport => "CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT",
    MatmulAlgorithmCapAttribute::StridedBatchSupport => "CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT",
    MatmulAlgorithmCapAttribute::OutOfPlaceResultSupport => "CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT",
    MatmulAlgorithmCapAttribute::UploSupport => "CUBLASLT_ALGO_CAP_UPLO_SUPPORT",
    MatmulAlgorithmCapAttribute::TileIds => "CUBLASLT_ALGO_CAP_TILE_IDS",
    MatmulAlgorithmCapAttribute::CustomOptionMax => "CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX",
    MatmulAlgorithmCapAttribute::CustomMemoryOrder => "CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER",
    MatmulAlgorithmCapAttribute::PointerModeMask => "CUBLASLT_ALGO_CAP_POINTER_MODE_MASK",
    MatmulAlgorithmCapAttribute::EpilogueMask => "CUBLASLT_ALGO_CAP_EPILOGUE_MASK",
    MatmulAlgorithmCapAttribute::StagesIds => "CUBLASLT_ALGO_CAP_STAGES_IDS",
    MatmulAlgorithmCapAttribute::LeadingDimensionNegative => "CUBLASLT_ALGO_CAP_LD_NEGATIVE",
    MatmulAlgorithmCapAttribute::NumericalImplementationFlags => "CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS",
    MatmulAlgorithmCapAttribute::MinAlignmentABytes => "CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES",
    MatmulAlgorithmCapAttribute::MinAlignmentBBytes => "CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES",
    MatmulAlgorithmCapAttribute::MinAlignmentCBytes => "CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES",
    MatmulAlgorithmCapAttribute::MinAlignmentDBytes => "CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES",
    MatmulAlgorithmCapAttribute::PointerArrayBatchSupport => "CUBLASLT_ALGO_CAP_POINTER_ARRAY_BATCH_SUPPORT",
    MatmulAlgorithmCapAttribute::FloatingPointEmulationSupport => "CUBLASLT_ALGO_CAP_FLOATING_POINT_EMULATION_SUPPORT",
    MatmulAlgorithmCapAttribute::PointerArrayGroupedSupport => "CUBLASLT_ALGO_CAP_POINTER_ARRAY_GROUPED_SUPPORT",
});

/// [`MatmulAlgorithmConfigAttribute`] identifies configurable attributes of a
/// [`MatmulAlgorithm`](crate::lt::matmul::MatmulAlgorithm).
/// Keep algorithm-specific configuration compatible with the reported capability
/// attributes.
/// Use [`MatmulAlgorithm::config_attribute`](crate::lt::matmul::MatmulAlgorithm::config_attribute)
/// and [`MatmulAlgorithm::set_config_attribute`](crate::lt::matmul::MatmulAlgorithm::set_config_attribute)
/// to query and update them.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum MatmulAlgorithmConfigAttribute {
    Id = sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_ID as _,
    TileId = sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_TILE_ID as _,
    SplitKCount = sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM as _,
    ReductionScheme =
        sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME as _,
    CtaSwizzling =
        sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING as _,
    CustomOption =
        sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION as _,
    StagesId = sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_STAGES_ID as _,
    InnerShapeId =
        sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID as _,
    ClusterShapeId =
        sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID as _,
}

impl_enum_conversion!(
    sys::cublasLtMatmulAlgoConfigAttributes_t,
    MatmulAlgorithmConfigAttribute
);

impl_enum_display!(MatmulAlgorithmConfigAttribute, {
    MatmulAlgorithmConfigAttribute::Id => "CUBLASLT_ALGO_CONFIG_ID",
    MatmulAlgorithmConfigAttribute::TileId => "CUBLASLT_ALGO_CONFIG_TILE_ID",
    MatmulAlgorithmConfigAttribute::SplitKCount => "CUBLASLT_ALGO_CONFIG_SPLITK_NUM",
    MatmulAlgorithmConfigAttribute::ReductionScheme => "CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME",
    MatmulAlgorithmConfigAttribute::CtaSwizzling => "CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING",
    MatmulAlgorithmConfigAttribute::CustomOption => "CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION",
    MatmulAlgorithmConfigAttribute::StagesId => "CUBLASLT_ALGO_CONFIG_STAGES_ID",
    MatmulAlgorithmConfigAttribute::InnerShapeId => "CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID",
    MatmulAlgorithmConfigAttribute::ClusterShapeId => "CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID",
});

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PointerModeMask: u32 {
        const HOST_POINTER = sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_HOST as _;
        const DEVICE_POINTER = sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_DEVICE as _;
        const DEVICE_VECTOR_POINTER =
            sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_DEVICE_VECTOR as _;
        const ALPHA_DEVICE_VECTOR_BETA_ZERO =
            sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO as _;
        const ALPHA_DEVICE_VECTOR_BETA_HOST =
            sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_HOST as _;
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NumericalImplFlags: u64 {
        const FMA = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA as _;
        const HMMA = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA as _;
        const IMMA = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_IMMA as _;
        const DMMA = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_DMMA as _;
        const TENSOR_OP_MASK = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK as _;
        const OP_TYPE_MASK = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_TYPE_MASK as _;
        const ACCUMULATOR_16F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F as _;
        const ACCUMULATOR_32F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F as _;
        const ACCUMULATOR_64F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_64F as _;
        const ACCUMULATOR_32I = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I as _;
        const ACCUMULATOR_TYPE_MASK =
            sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK as _;
        const INPUT_16F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F as _;
        const INPUT_16BF = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF as _;
        const INPUT_TF32 = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32 as _;
        const INPUT_32F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_32F as _;
        const INPUT_64F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_64F as _;
        const INPUT_8I = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8I as _;
        const INPUT_8F_E4M3 = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E4M3 as _;
        const INPUT_8F_E5M2 = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E5M2 as _;
        const OP_INPUT_TYPE_MASK = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK as _;
        const GAUSSIAN = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_GAUSSIAN as _;
    }
}
