use std::fmt::{self, Display, Formatter};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cublas_sys as sys;

singe_core::native_enum! {
    /// Matrix data ordering.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum Order: sys::cublasLtOrder_t {
        /// Data is ordered in column-major format.
        /// The leading dimension is the stride (in elements) to the beginning of next column in memory.
        ColumnMajor = CUBLASLT_ORDER_COL,
        /// Data is ordered in row-major format.
        /// The leading dimension is the stride (in elements) to the beginning of next row in memory.
        RowMajor = CUBLASLT_ORDER_ROW,
        /// Column-major tiles of 32 columns.
        /// The leading dimension is the stride, in elements, to the start of the next group of 32 columns.
        /// A matrix with 33 columns and 2 rows needs a leading dimension of at least `32 * 2 = 64`.
        Column32 = CUBLASLT_ORDER_COL32,
        /// Column-major composite tiles of 32 columns and 8 rows.
        /// Each tile interleaves inner tiles of 4 columns within 4 even or odd rows, alternating.
        /// The leading dimension is the stride, in elements, to the first 32x8 tile of the next group of 32 columns.
        /// A matrix with 33 columns and 1 row needs a leading dimension of at least `(32 * 8) * 1 = 256`.
        Column4_4R2_8C = CUBLASLT_ORDER_COL4_4R2_8C,
        /// Column-major composite tiles of 32 columns and 32 rows.
        /// The element offset within a tile is `(((row % 8) / 2 * 4 + row / 8) * 2 + row % 2) * 32 + col`.
        /// The leading dimension is the stride, in elements, to the first 32x32 tile of the next group of 32 columns.
        /// A matrix with 33 columns and 1 row needs a leading dimension of at least `(32 * 32) * 1 = 1024`.
        Column32TwoRowsFourRowsFourColumns = CUBLASLT_ORDER_COL32_2R_4R4,
    }
}

impl_enum_display!(Order, {
    Order::ColumnMajor => "CUBLASLT_ORDER_COL",
    Order::RowMajor => "CUBLASLT_ORDER_ROW",
    Order::Column32 => "CUBLASLT_ORDER_COL32",
    Order::Column4_4R2_8C => "CUBLASLT_ORDER_COL4_4R2_8C",
    Order::Column32TwoRowsFourRowsFourColumns => "CUBLASLT_ORDER_COL32_2R_4R4",
});

singe_core::native_enum! {
    /// Batch processing mode for matrix layouts.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum BatchMode: sys::cublasLtBatchMode_t {
        /// The matrices of each instance of the batch are located at fixed offsets in number of elements from their locations in the previous instance.
        Strided = CUBLASLT_BATCH_MODE_STRIDED,
        /// The address of the matrix of each instance of the batch are read from device arrays of pointers.
        PointerArray = CUBLASLT_BATCH_MODE_POINTER_ARRAY,
        /// Experimental: each matrix of the group is read through device arrays of pointers.
        /// Each group can have different rows, columns, and leading dimensions; see [`MatrixLayout::create_grouped`](crate::lt::descriptor::MatrixLayout::create_grouped).
        Grouped = CUBLASLT_BATCH_MODE_GROUPED,
    }
}

impl_enum_display!(BatchMode, {
    BatchMode::Strided => "CUBLASLT_BATCH_MODE_STRIDED",
    BatchMode::PointerArray => "CUBLASLT_BATCH_MODE_POINTER_ARRAY",
    BatchMode::Grouped => "CUBLASLT_BATCH_MODE_GROUPED",
});

singe_core::native_enum! {
    /// Experimental integer width for grouped-matrix dimension arrays.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum IntegerWidth: sys::cublasLtIntegerWidth_t {
        /// 32-bit integers.
        Bits32 = CUBLASLT_INTEGER_WIDTH_32,
        /// 64-bit integers.
        Bits64 = CUBLASLT_INTEGER_WIDTH_64,
    }
}

impl_enum_display!(IntegerWidth, {
    IntegerWidth::Bits32 => "CUBLASLT_INTEGER_WIDTH_32",
    IntegerWidth::Bits64 => "CUBLASLT_INTEGER_WIDTH_64",
});

singe_core::native_enum! {
    /// [`MatrixLayoutAttribute`] identifies the configurable attributes of a [`MatrixLayout`](crate::lt::descriptor::MatrixLayout).
    /// Use [`MatrixLayout::attribute`](crate::lt::descriptor::MatrixLayout::attribute) and [`MatrixLayout::set_attribute`](crate::lt::descriptor::MatrixLayout::set_attribute) to query and update them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum MatrixLayoutAttribute: sys::cublasLtMatrixLayoutAttribute_t {
        /// Element data type, a [`DataType`](singe_cuda::data_type::DataType).
        Type = CUBLASLT_MATRIX_LAYOUT_TYPE,
        /// Memory order of the matrix data, an [`Order`]; the default is [`Order::ColumnMajor`].
        Order = CUBLASLT_MATRIX_LAYOUT_ORDER,
        /// Number of rows; values beyond the `i32` range are normally unsupported.
        Rows = CUBLASLT_MATRIX_LAYOUT_ROWS,
        /// Number of columns; values beyond the `i32` range are normally unsupported.
        Cols = CUBLASLT_MATRIX_LAYOUT_COLS,
        /// Leading dimension; for [`Order::ColumnMajor`] it is the stride, in elements, between columns.
        /// Only non-negative values are supported.
        /// It must be large enough that columns do not overlap, such as at least [`MatrixLayoutAttribute::Rows`] for [`Order::ColumnMajor`].
        LeadingDimension = CUBLASLT_MATRIX_LAYOUT_LD,
        /// Number of matrix multiplications in the batch; the default is 1.
        /// The batch support capabilities, such as [`MatmulAlgorithmCapAttribute::StridedBatchSupport`], say which batch modes an algorithm accepts.
        BatchCount = CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
        /// Stride, in elements, to the next matrix of a strided batch; the default is 0.
        /// For a planar-complex layout, [`matmul_async`](crate::lt::matmul::matmul_async) counts this stride in real sub-elements, so a 1024-byte offset of [`DataType::ComplexF16`](singe_cuda::data_type::DataType::ComplexF16) data is a stride of 512.
        /// [`matrix_transform_async`](crate::lt::matmul::matrix_transform_async) counts it in complex elements instead, so the same offset is a stride of 256 there.
        StridedBatchOffset = CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
        /// Stride, in bytes, to the imaginary plane of a planar-complex layout.
        /// The default of 0 stores real and imaginary parts interleaved.
        PlaneOffset = CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET,
        /// Batch mode of the matrix, a [`BatchMode`]; the default is [`BatchMode::Strided`].
        BatchMode = CUBLASLT_MATRIX_LAYOUT_BATCH_MODE,
        /// Experimental: device address of the per-group row counts of a grouped matrix.
        /// The array holds [`MatrixLayoutAttribute::BatchCount`] elements of the width set by [`MatrixLayoutAttribute::GroupedRowsColsArrayIntegerWidth`]; the default is null.
        GroupedRowsArray = CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_ARRAY,
        /// Experimental: device address of the per-group column counts of a grouped matrix.
        /// The array holds [`MatrixLayoutAttribute::BatchCount`] elements of the width set by [`MatrixLayoutAttribute::GroupedRowsColsArrayIntegerWidth`]; the default is null.
        GroupedColsArray = CUBLASLT_GROUPED_MATRIX_LAYOUT_COLS_ARRAY,
        /// Experimental: device address of the per-group leading dimensions of a grouped matrix.
        /// The array holds [`MatrixLayoutAttribute::BatchCount`] elements of the width set by [`MatrixLayoutAttribute::GroupedLeadingDimensionArrayIntegerWidth`]; the default is null.
        GroupedLeadingDimensionArray = CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY,
        /// Experimental integer width for grouped row and column arrays.
        ///
        /// Stored as `i32`, with [`IntegerWidth::Bits32`] as the default.
        GroupedRowsColsArrayIntegerWidth = CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH,
        /// Experimental integer width for grouped leading-dimension arrays.
        ///
        /// Stored as `i32`, with [`IntegerWidth::Bits32`] as the default.
        GroupedLeadingDimensionArrayIntegerWidth = CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH,
    }
}

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

singe_core::native_enum! {
    /// Pointer mode for the scaling factors `alpha` and `beta`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum PointerMode: sys::cublasLtPointerMode_t {
        /// `alpha` and `beta` are single values in host memory, as with [`crate::types::PointerMode::Host`].
        Host = CUBLASLT_POINTER_MODE_HOST,
        /// `alpha` and `beta` are single values in device memory, as with [`crate::types::PointerMode::Device`].
        Device = CUBLASLT_POINTER_MODE_DEVICE,
        /// `alpha` and `beta` are arrays in device memory.
        DeviceVector = CUBLASLT_POINTER_MODE_DEVICE_VECTOR,
        /// `alpha` is an array in device memory and `beta` is zero.
        /// [`MatmulDescriptorAttribute::AlphaVectorBatchStride`] must be 0 in this mode.
        AlphaDeviceVectorBetaZero = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO,
        /// `alpha` is an array in device memory and `beta` is a single value in host memory.
        AlphaDeviceVectorBetaHost = CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST,
    }
}

impl_enum_display!(PointerMode, {
    PointerMode::Host => "CUBLASLT_POINTER_MODE_HOST",
    PointerMode::Device => "CUBLASLT_POINTER_MODE_DEVICE",
    PointerMode::DeviceVector => "CUBLASLT_POINTER_MODE_DEVICE_VECTOR",
    PointerMode::AlphaDeviceVectorBetaZero => "CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO",
    PointerMode::AlphaDeviceVectorBetaHost => "CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST",
});

singe_core::native_enum! {
    /// [`MatmulDescriptorAttribute`] identifies the configurable attributes of a [`MatmulDescriptor`](crate::lt::matmul::MatmulDescriptor).
    /// Use [`MatmulDescriptor::attribute`](crate::lt::matmul::MatmulDescriptor::attribute) and [`MatmulDescriptor::set_attribute`](crate::lt::matmul::MatmulDescriptor::set_attribute) to query and update them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum MatmulDescriptorAttribute: sys::cublasLtMatmulDescAttributes_t {
        /// Requests batch-invariant matrix multiplication results.
        BatchInvarianceFlags = CUBLASLT_MATMUL_DESC_BATCH_INVARIANCE_FLAGS,
        /// Type used for multiplication and accumulation, a [`ComputeType`](crate::types::ComputeType).
        ComputeType = CUBLASLT_MATMUL_DESC_COMPUTE_TYPE,
        /// Data type of `alpha` and `beta`, a [`DataType`](singe_cuda::data_type::DataType).
        /// The accumulator and `C` are converted to this type before scaling, and the result is converted to the type of `D` before it is stored.
        /// The default depends on [`MatmulDescriptorAttribute::ComputeType`].
        ScaleType = CUBLASLT_MATMUL_DESC_SCALE_TYPE,
        /// Where `alpha` and `beta` live and whether they are vectors, a [`PointerMode`]; the default is [`PointerMode::Host`].
        PointerMode = CUBLASLT_MATMUL_DESC_POINTER_MODE,
        /// Operation applied to `A`, an [`Operation`](crate::types::Operation); the default is [`Operation::NonTranspose`](crate::types::Operation::NonTranspose).
        TransposeA = CUBLASLT_MATMUL_DESC_TRANSA,
        /// Operation applied to `B`, an [`Operation`](crate::types::Operation); the default is [`Operation::NonTranspose`](crate::types::Operation::NonTranspose).
        TransposeB = CUBLASLT_MATMUL_DESC_TRANSB,
        /// Operation applied to `C`, an [`Operation`](crate::types::Operation); only the default, [`Operation::NonTranspose`](crate::types::Operation::NonTranspose), is supported.
        TransposeC = CUBLASLT_MATMUL_DESC_TRANSC,
        /// Which triangle of the output is filled and used, a [`FillMode`](crate::types::FillMode); the default is [`FillMode::Full`](crate::types::FillMode::Full).
        /// Other values are unsupported for bfloat16 and FP8 data and on Hopper and Blackwell GPUs.
        FillMode = CUBLASLT_MATMUL_DESC_FILL_MODE,
        /// Post-processing applied to the result, an [`Epilogue`]; the default is [`Epilogue::Default`].
        Epilogue = CUBLASLT_MATMUL_DESC_EPILOGUE,
        /// Device address of the bias or bias gradient vector; the default is null.
        /// It is an input with one element per row of `D` for [`Epilogue::Bias`], [`Epilogue::ReluBias`], [`Epilogue::ReluAuxBias`], [`Epilogue::GeluBias`], and [`Epilogue::GeluAuxBias`].
        /// It is an output with one element per row of `D` for [`Epilogue::DReluBGrad`], [`Epilogue::DGeluBGrad`], and [`Epilogue::BGradA`], and one element per column of `D` for [`Epilogue::BGradB`].
        /// Its elements have the scale type when `D` is [`DataType::I8`](singe_cuda::data_type::DataType::I8) and the type of `D` otherwise, unless [`MatmulDescriptorAttribute::BiasDataType`] is set.
        BiasPointer = CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        /// Stride, in elements, to the next bias or bias gradient vector of a strided batch; the default is 0.
        /// It must be 1 when a matrix uses [`BatchMode::Grouped`] and the epilogue includes [`Epilogue::Bias`].
        BiasBatchStride = CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE,
        /// Device address of the epilogue's auxiliary buffer, whose type is [`MatmulDescriptorAttribute::EpilogueAuxDataType`].
        /// It receives the `ReLU` bit mask for [`Epilogue::ReluAux`] and [`Epilogue::ReluAuxBias`], and provides it for [`Epilogue::DRelu`] and [`Epilogue::DReluBGrad`].
        /// It receives the GELU input for [`Epilogue::GeluAux`] and [`Epilogue::GeluAuxBias`], and provides it for [`Epilogue::DGelu`] and [`Epilogue::DGeluBGrad`].
        /// Heuristic queries read the address to determine its alignment, and it requires [`MatmulDescriptorAttribute::EpilogueAuxLeadingDimension`].
        EpilogueAuxPointer = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
        /// Leading dimension, in elements, of the epilogue's auxiliary buffer.
        /// For a `ReLU` bit mask it counts bits and must be a multiple of 128 no smaller than the rows of the output.
        /// For a GELU input it must be a multiple of 8 no smaller than the rows of the output.
        EpilogueAuxLeadingDimension = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
        /// Batch stride, in elements, of the epilogue's auxiliary buffer; the default is 0.
        /// For a `ReLU` bit mask it counts bits and must be a multiple of 128; for a GELU input it must be a multiple of 8.
        EpilogueAuxBatchStride = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
        /// Batch stride of the `alpha` vector, used with [`PointerMode::AlphaDeviceVectorBetaHost`] when `D` has a [`MatrixLayoutAttribute::BatchCount`] above 1; the default is 0.
        /// It must be 0 with [`PointerMode::AlphaDeviceVectorBetaZero`] and when any matrix uses a batch mode other than [`BatchMode::Strided`].
        AlphaVectorBatchStride = CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE,
        /// Number of SMs the heuristics target, for when concurrent work occupies part of the device; the default of 0 targets all of them.
        SmCountTarget = CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
        /// Device address of the factor that scales `A` into the compute type's range, of the compute type; the default is null, which means 1.
        /// An unsupported combination of data, scale, and compute types makes [`matmul_async`](crate::lt::matmul::matmul_async) return [`Status::InvalidValue`](crate::error::Status::InvalidValue).
        AScalePointer = CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        /// Same as [`MatmulDescriptorAttribute::AScalePointer`], for `B`.
        BScalePointer = CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        /// Same as [`MatmulDescriptorAttribute::AScalePointer`], for `C`.
        CScalePointer = CUBLASLT_MATMUL_DESC_C_SCALE_POINTER,
        /// Same as [`MatmulDescriptorAttribute::AScalePointer`], for `D`.
        DScalePointer = CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
        /// Device address that receives the maximum absolute value of the output, of the compute type; the default is null, which skips the computation.
        /// An unsupported combination of data, scale, and compute types makes [`matmul_async`](crate::lt::matmul::matmul_async) return [`Status::InvalidValue`](crate::error::Status::InvalidValue).
        AmaxDPointer = CUBLASLT_MATMUL_DESC_AMAX_D_POINTER,
        /// Data type stored through [`MatmulDescriptorAttribute::EpilogueAuxPointer`]; the default of -1 selects the type of `D`, and a `ReLU` bit mask is always bits.
        /// FP8 kernels with a [`DataType::F8E4M3`](singe_cuda::data_type::DataType::F8E4M3) output, FP8 E4M3 inputs, [`DataType::F16`](singe_cuda::data_type::DataType::F16) bias, and [`Epilogue::GeluAux`] accept another type.
        /// With a [`DataType::F16`](singe_cuda::data_type::DataType::F16) `C`, it may be F16 or F8 E4M3; with a [`DataType::Bf16`](singe_cuda::data_type::DataType::Bf16) `C`, it may be BF16.
        /// An unsupported combination makes [`matmul_async`](crate::lt::matmul::matmul_async) return [`Status::InvalidValue`](crate::error::Status::InvalidValue).
        EpilogueAuxDataType = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
        /// Device address of the factor that scales results into the storage range of the auxiliary buffer, of the compute type; the default is null, which means 1.
        /// An unsupported combination of data, scale, and compute types makes [`matmul_async`](crate::lt::matmul::matmul_async) return [`Status::InvalidValue`](crate::error::Status::InvalidValue).
        EpilogueAuxScalePointer = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
        /// Device address that receives the maximum absolute value of the auxiliary buffer, of the compute type; the default is null, which skips the computation.
        /// An unsupported combination of data, scale, and compute types makes [`matmul_async`](crate::lt::matmul::matmul_async) return [`Status::InvalidValue`](crate::error::Status::InvalidValue).
        EpilogueAuxAmaxPointer = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER,
        /// Nonzero enables FP8 fast accumulation, which skips periodic promotion of intermediate results to a higher precision; the default is 0.
        /// It affects Ada and Hopper GPUs.
        FastAccum = CUBLASLT_MATMUL_DESC_FAST_ACCUM,
        /// Data type of the bias or bias gradient vector; the default of -1 selects the type of `D`.
        /// IMMA kernels with an [`DataType::I32`](singe_cuda::data_type::DataType::I32) compute type and an [`DataType::I8`](singe_cuda::data_type::DataType::I8) `C` use the scale type instead.
        /// FP8 kernels with an [`DataType::F32`](singe_cuda::data_type::DataType::F32), [`DataType::F8E4M3`](singe_cuda::data_type::DataType::F8E4M3), or [`DataType::F8E5M2`](singe_cuda::data_type::DataType::F8E5M2) output are also exceptions.
        BiasDataType = CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
        /// How the scale factor of `A` is interpreted, a [`MatrixScale`]; the default is 0.
        AScaleMode = CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
        /// How the scale factor of `B` is interpreted, a [`MatrixScale`]; the default is 0.
        BScaleMode = CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
        /// How the scale factor of `C` is interpreted, a [`MatrixScale`]; the default is 0.
        CScaleMode = CUBLASLT_MATMUL_DESC_C_SCALE_MODE,
        /// How the scale factor of `D` is interpreted, a [`MatrixScale`]; the default is 0.
        DScaleMode = CUBLASLT_MATMUL_DESC_D_SCALE_MODE,
        /// How the scale factor of the auxiliary buffer is interpreted, a [`MatrixScale`]; the default is 0.
        EpilogueAuxScaleMode = CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE,
        /// Device address of the factors that scale `D` into the compute type's range, typed by [`MatmulDescriptorAttribute::DOutScaleMode`]; the default is null.
        /// Setting it for an unsupported combination, or leaving it unset for a combination that needs it, makes [`matmul_async`](crate::lt::matmul::matmul_async) return [`Status::InvalidValue`](crate::error::Status::InvalidValue).
        DOutScalePointer = CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER,
        /// How the output scale factor of `D` is interpreted, a [`MatrixScale`]; the default is 0.
        DOutScaleMode = CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE,
        /// Emulation descriptor configuring floating-point emulation; the default is null.
        EmulationDescriptor = CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR,
        /// Experimental: 1 makes `alpha` a device array of [`MatrixLayoutAttribute::BatchCount`] pointers, one per batch; the default is 0.
        /// It applies when `D` has a batch count above 1, every matrix uses [`BatchMode::Grouped`], and the pointer mode is [`PointerMode::Device`].
        AlphaBatchStride = CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE,
        /// Experimental: 1 makes `beta` a device array of [`MatrixLayoutAttribute::BatchCount`] pointers, one per batch; the default is 0.
        /// It applies when `D` has a batch count above 1, every matrix uses [`BatchMode::Grouped`], and the pointer mode is [`PointerMode::Device`].
        BetaBatchStride = CUBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE,
    }
}

impl_enum_display!(MatmulDescriptorAttribute, {
    MatmulDescriptorAttribute::BatchInvarianceFlags => "CUBLASLT_MATMUL_DESC_BATCH_INVARIANCE_FLAGS",
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

singe_core::native_enum! {
    /// Scaling mode that defines how scaling factor pointers are interpreted.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum MatrixScale: sys::cublasLtMatmulMatrixScale_t {
        /// Packed UE8M0 scale factors in a column-major layout.
        /// Four exponents covering consecutive 128-element K blocks share one `u32`.
        /// The `u32` elements are stored M-major for `A` and N-major for `B`, with shape `round_up(M, 4)` x `ceil(K / 512)` or `round_up(N, 4)` x `ceil(K / 512)`.
        Vector128MnK4UE8M0 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_MN_K4_UE8M0,
        /// Same as [`Self::Vector128MnK4UE8M0`] with 32-element K blocks.
        /// The shape is `round_up(M, 4)` x `ceil(K / 128)` for `A` or `round_up(N, 4)` x `ceil(K / 128)` for `B`.
        Vector32MnK4UE8M0 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_MN_K4_UE8M0,
        /// Scaling factors are single precision scalars applied to the whole tensor.
        Scalar32F = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F,
        /// `f32` vectors of M elements for `A` and N elements for `B`, valid only for those two matrices.
        /// Element `(i, j)` of the product is multiplied by element `i` of the `A` scale and element `j` of the `B` scale.
        OuterVector32F = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F,
        /// Same as [`Self::Vector16UE4M3`], except that scaling factor tensor elements have type UE8M0 and the block size is 32 elements.
        Vector32UE8M0 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0,
        /// Scaling factors are tensors that contain a dedicated scaling factor stored as an 8-bit UE4M3 value for each 16-element block in the innermost dimension of the corresponding data tensor.
        Vector16UE4M3 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
        /// Scaling factors are tensors that contain a dedicated `f32` scaling factor for each 128-element block in the innermost dimension of the corresponding data tensor.
        Vector12832F = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F,
        /// Scaling factors are tensors that contain a dedicated `f32` scaling factor for each 128x128-element block in the the corresponding data tensor.
        Block128x12832F = CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
        /// Experimental: Scaling factors are single precision scalars applied to the whole tensor.
        /// Each batch has its own scaling factor.
        PerBatchScalar32F = CUBLASLT_MATMUL_MATRIX_SCALE_PER_BATCH_SCALAR_32F,
    }
}

impl_enum_display!(MatrixScale, {
    MatrixScale::Vector128MnK4UE8M0 => "CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_MN_K4_UE8M0",
    MatrixScale::Vector32MnK4UE8M0 => "CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_MN_K4_UE8M0",
    MatrixScale::Scalar32F => "CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F",
    MatrixScale::OuterVector32F => "CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F",
    MatrixScale::Vector32UE8M0 => "CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0",
    MatrixScale::Vector16UE4M3 => "CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3",
    MatrixScale::Vector12832F => "CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F",
    MatrixScale::Block128x12832F => "CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F",
    MatrixScale::PerBatchScalar32F => "CUBLASLT_MATMUL_MATRIX_SCALE_PER_BATCH_SCALAR_32F",
});

singe_core::native_enum! {
    /// [`MatrixTransformDescAttribute`] identifies the configurable attributes of a [`MatrixTransformDescriptor`](crate::lt::descriptor::MatrixTransformDescriptor).
    /// Use [`MatrixTransformDescriptor::attribute`](crate::lt::descriptor::MatrixTransformDescriptor::attribute) and [`MatrixTransformDescriptor::set_attribute`](crate::lt::descriptor::MatrixTransformDescriptor::set_attribute) to query and update them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum MatrixTransformDescAttribute: sys::cublasLtMatrixTransformDescAttributes_t {
        /// Data type used for scaling and summation, a [`DataType`](singe_cuda::data_type::DataType).
        /// Inputs are converted to it, and results are converted to the output type before they are stored.
        ScaleType = CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE,
        /// Whether `alpha` and `beta` are in host or device memory, a [`PointerMode`]; the default is [`PointerMode::Host`].
        PointerMode = CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
        /// Operation applied to `A`, an [`Operation`](crate::types::Operation); the default is [`Operation::NonTranspose`](crate::types::Operation::NonTranspose).
        TransposeA = CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA,
        /// Operation applied to `B`, an [`Operation`](crate::types::Operation); the default is [`Operation::NonTranspose`](crate::types::Operation::NonTranspose).
        TransposeB = CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB,
    }
}

impl_enum_display!(MatrixTransformDescAttribute, {
    MatrixTransformDescAttribute::ScaleType => "CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE",
    MatrixTransformDescAttribute::PointerMode => "CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE",
    MatrixTransformDescAttribute::TransposeA => "CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA",
    MatrixTransformDescAttribute::TransposeB => "CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSB",
});

singe_core::native_enum! {
    /// [`EmulationDescAttribute`] identifies the configurable floating-point emulation parameters of an emulation descriptor.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum EmulationDescAttribute: sys::cublasLtEmulationDescAttributes_t {
        /// When to use floating-point emulation, an [`EmulationStrategy`](crate::types::EmulationStrategy); the default is [`EmulationStrategy::Default`](crate::types::EmulationStrategy::Default).
        Strategy = CUBLASLT_EMULATION_DESC_STRATEGY,
        /// Special floating-point values the emulation must handle, an [`EmulationSpecialValuesSupport`](singe_cuda::types::EmulationSpecialValuesSupport).
        SpecialValuesSupport = CUBLASLT_EMULATION_DESC_SPECIAL_VALUES_SUPPORT,
        /// Selects dynamic or fixed mantissa control for fixed-point emulation.
        ///
        /// Stored as `i32`, using the values of [`EmulationMantissaControl`](singe_cuda::types::EmulationMantissaControl); the default value is zero.
        /// Dynamic mode uses the recommended bit count plus the configured offset, subject to quantization and the maximum bit count.
        /// Dynamic mode falls back to floating-point computation when the quantized requirement exceeds the quantized maximum.
        /// Fixed mode uses the quantized maximum bit count.
        FixedPointMantissaControl = CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL,
        /// Maximum retained mantissa bit count for fixed-point emulation, subject to quantization.
        ///
        /// Stored as `i32`; the default of zero lets the library choose a device-dependent limit.
        FixedPointMaxMantissaBitCount = CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT,
        /// Bias added to the recommended mantissa bit count in dynamic mode.
        ///
        /// Stored as `i32`, with a default of zero.
        FixedPointMantissaBitOffset = CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_OFFSET,
        /// Device pointer to an `i32` receiving the actual retained mantissa bit count.
        ///
        /// The library writes `-1` when fixed-point emulation is not used.
        /// The default pointer is null.
        FixedPointMantissaBitCountPointer = CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_COUNT_POINTER,
    }
}

impl_enum_display!(EmulationDescAttribute, {
    EmulationDescAttribute::Strategy => "CUBLASLT_EMULATION_DESC_STRATEGY",
    EmulationDescAttribute::SpecialValuesSupport => "CUBLASLT_EMULATION_DESC_SPECIAL_VALUES_SUPPORT",
    EmulationDescAttribute::FixedPointMantissaControl => "CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL",
    EmulationDescAttribute::FixedPointMaxMantissaBitCount => "CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT",
    EmulationDescAttribute::FixedPointMantissaBitOffset => "CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_OFFSET",
    EmulationDescAttribute::FixedPointMantissaBitCountPointer => "CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_COUNT_POINTER",
});

singe_core::native_enum! {
    /// [`ReductionScheme`] specifies a reduction scheme for the portions of the dot product calculated in parallel, also known as split-K.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum ReductionScheme: sys::cublasLtReductionScheme_t {
        /// No split-K reduction; the dot product runs in one sequence.
        None = CUBLASLT_REDUCTION_SCHEME_NONE,
        /// Parts are added up in place in the output buffer, in the output type.
        /// The workspace holds only the counters that order the additions.
        InPlace = CUBLASLT_REDUCTION_SCHEME_INPLACE,
        /// Intermediate results are stored in compute type in the workspace and reduced in a separate step.
        ComputeType = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE,
        /// Intermediate results are stored in output type in the workspace and reduced in a separate step.
        OutputType = CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE,
        /// Mask of every reduction scheme.
        Mask = CUBLASLT_REDUCTION_SCHEME_MASK,
    }
}

impl_enum_display!(ReductionScheme, {
    ReductionScheme::None => "CUBLASLT_REDUCTION_SCHEME_NONE",
    ReductionScheme::InPlace => "CUBLASLT_REDUCTION_SCHEME_INPLACE",
    ReductionScheme::ComputeType => "CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE",
    ReductionScheme::OutputType => "CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE",
    ReductionScheme::Mask => "CUBLASLT_REDUCTION_SCHEME_MASK",
});

singe_core::native_enum! {
    /// [`Epilogue`] selects post-processing applied after the main matrix multiplication.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum Epilogue: sys::cublasLtEpilogue_t {
        /// No post-processing beyond scaling and quantizing the result.
        Default = CUBLASLT_EPILOGUE_DEFAULT,
        /// Applies `ReLU`, `x = max(x, 0)`, to the result.
        Relu = CUBLASLT_EPILOGUE_RELU,
        /// Applies `ReLU` and writes its bit mask to [`MatmulDescriptorAttribute::EpilogueAuxPointer`].
        ReluAux = CUBLASLT_EPILOGUE_RELU_AUX,
        /// Adds the bias vector, broadcast across columns, before the final post-processing.
        /// The vector is packed and has one element per row of `D`.
        Bias = CUBLASLT_EPILOGUE_BIAS,
        /// Adds the bias, then applies `ReLU`.
        ReluBias = CUBLASLT_EPILOGUE_RELU_BIAS,
        /// Adds the bias, applies `ReLU`, and writes the `ReLU` bit mask to [`MatmulDescriptorAttribute::EpilogueAuxPointer`].
        ReluAuxBias = CUBLASLT_EPILOGUE_RELU_AUX_BIAS,
        /// Applies the `ReLU` gradient to the product and stores it in the output.
        /// It reads the bit mask from [`MatmulDescriptorAttribute::EpilogueAuxPointer`].
        DRelu = CUBLASLT_EPILOGUE_DRELU,
        /// Applies the `ReLU` gradient to the product, stores it in the output, and writes the bias gradient to [`MatmulDescriptorAttribute::BiasPointer`].
        /// It reads the bit mask from [`MatmulDescriptorAttribute::EpilogueAuxPointer`].
        DReluBGrad = CUBLASLT_EPILOGUE_DRELU_BGRAD,
        /// Applies GELU to the result.
        Gelu = CUBLASLT_EPILOGUE_GELU,
        /// Applies GELU and writes its input to [`MatmulDescriptorAttribute::EpilogueAuxPointer`].
        GeluAux = CUBLASLT_EPILOGUE_GELU_AUX,
        /// Adds the bias, then applies GELU.
        GeluBias = CUBLASLT_EPILOGUE_GELU_BIAS,
        /// Adds the bias, applies GELU, and writes the GELU input to [`MatmulDescriptorAttribute::EpilogueAuxPointer`].
        GeluAuxBias = CUBLASLT_EPILOGUE_GELU_AUX_BIAS,
        /// Applies the GELU gradient to the product and stores it in the output.
        /// It reads the GELU input from [`MatmulDescriptorAttribute::EpilogueAuxPointer`].
        DGelu = CUBLASLT_EPILOGUE_DGELU,
        /// Applies the GELU gradient to the product, stores it in the output, and writes the bias gradient to [`MatmulDescriptorAttribute::BiasPointer`].
        /// It reads the GELU input from [`MatmulDescriptorAttribute::EpilogueAuxPointer`].
        DGeluBGrad = CUBLASLT_EPILOGUE_DGELU_BGRAD,
        /// Writes the bias gradient of `A`, reduced over K with one element per row of `D`, to [`MatmulDescriptorAttribute::BiasPointer`].
        BGradA = CUBLASLT_EPILOGUE_BGRADA,
        /// Writes the bias gradient of `B`, reduced over K with one element per column of `D`, to [`MatmulDescriptorAttribute::BiasPointer`].
        BGradB = CUBLASLT_EPILOGUE_BGRADB,
    }
}

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

singe_core::native_enum! {
    /// Heuristics search mode.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum SearchMode: sys::cublasLtMatmulSearch_t {
        /// Searches for the best algorithm for the problem.
        BestFit = CUBLASLT_SEARCH_BEST_FIT,
        /// Searches only for the best configuration of a preconfigured algorithm ID.
        LimitedByAlgorithmId = CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID,
    }
}

impl_enum_display!(SearchMode, {
    SearchMode::BestFit => "CUBLASLT_SEARCH_BEST_FIT",
    SearchMode::LimitedByAlgorithmId => "CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID",
});

singe_core::native_enum! {
    /// [`MatmulPreferenceAttribute`] identifies algorithm-search preferences stored in a [`MatmulPreference`](crate::lt::matmul::MatmulPreference).
    /// Use [`MatmulPreference::attribute`](crate::lt::matmul::MatmulPreference::attribute) and [`MatmulPreference::set_attribute`](crate::lt::matmul::MatmulPreference::set_attribute) to query and update them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum MatmulPreferenceAttribute: sys::cublasLtMatmulPreferenceAttributes_t {
        /// Search mode.
        /// See [`SearchMode`].
        /// Default is [`SearchMode::BestFit`].
        SearchMode = CUBLASLT_MATMUL_PREF_SEARCH_MODE,
        /// Maximum allowed workspace memory.
        /// Default is 0 (no workspace memory allowed).
        MaxWorkspaceBytes = CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        /// Mask of the [`ReductionScheme`] values that algorithm configurations may use; the default, [`ReductionScheme::Mask`], allows all of them.
        ReductionSchemeMask = CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
        /// Guaranteed alignment of `A`, in bytes; the default is 256.
        /// Smaller values exclude algorithms that need stricter alignment.
        MinAlignmentABytes = CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
        /// Guaranteed alignment of `B`, in bytes; the default is 256.
        /// Smaller values exclude algorithms that need stricter alignment.
        MinAlignmentBBytes = CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        /// Guaranteed alignment of `C`, in bytes; the default is 256.
        /// Smaller values exclude algorithms that need stricter alignment.
        MinAlignmentCBytes = CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
        /// Guaranteed alignment of `D`, in bytes; the default is 256.
        /// Smaller values exclude algorithms that need stricter alignment.
        MinAlignmentDBytes = CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
        /// Maximum wave count, an `f32`; the default of 0 disables the limit.
        /// A nonzero value excludes algorithms whose [`MatmulHeuristicResult::waves_count`](crate::lt::matmul::MatmulHeuristicResult::waves_count) is higher.
        MaxWavesCount = CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT,
        /// Mask of the [`NumericalImplFlags`] that results may use; the default allows all of them.
        ImplementationMask = CUBLASLT_MATMUL_PREF_IMPL_MASK,
        /// Experimental average reduction dimension for grouped matrix multiplication.
        ///
        /// Stored as `u32`, with a default of zero.
        GroupedAverageReductionDim = CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM,
        /// Experimental average number of rows in grouped output matrices `D`.
        ///
        /// Stored as `u32`, with a default of zero.
        GroupedDescDAverageRows = CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS,
        /// Experimental average number of columns in grouped output matrices `D`.
        ///
        /// Stored as `u32`, with a default of zero.
        GroupedDescDAverageCols = CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS,
    }
}

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

singe_core::native_enum! {
    /// [`MatmulAlgorithmCapAttribute`] enumerates capability attributes available on an initialized [`MatmulAlgorithm`](crate::lt::matmul::MatmulAlgorithm) with [`MatmulAlgorithm::cap_attribute`](crate::lt::matmul::MatmulAlgorithm::cap_attribute).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum MatmulAlgorithmCapAttribute: sys::cublasLtMatmulAlgoCapAttributes_t {
        /// Nonzero when the algorithm supports split-K; see [`MatmulAlgorithmConfigAttribute::SplitKCount`].
        SplitKSupport = CUBLASLT_ALGO_CAP_SPLITK_SUPPORT,
        /// Mask of the supported [`ReductionScheme`] values.
        ReductionSchemeMask = CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK,
        /// 1 when the algorithm supports CTA swizzling, 0 when it does not; see [`MatmulAlgorithmConfigAttribute::CtaSwizzling`].
        CtaSwizzlingSupport = CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,
        /// Nonzero when the algorithm supports strided batches.
        StridedBatchSupport = CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT,
        /// Nonzero when `D` may differ from `C`.
        OutOfPlaceResultSupport = CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT,
        /// Nonzero when the algorithm supports symmetric and Hermitian rank-k updates.
        UploSupport = CUBLASLT_ALGO_CAP_UPLO_SUPPORT,
        /// Array of the tile IDs the algorithm accepts for [`MatmulAlgorithmConfigAttribute::TileId`].
        /// Query it with a zero-sized buffer to get the count; an algorithm without tile IDs reports only the undefined tile, 0.
        TileIds = CUBLASLT_ALGO_CAP_TILE_IDS,
        /// Largest value, inclusive, of [`MatmulAlgorithmConfigAttribute::CustomOption`].
        CustomOptionMax = CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX,
        /// Nonzero when the algorithm may require an [`Order`] other than column-major or row-major.
        CustomMemoryOrder = CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER,
        /// Bitmask enumerating the pointer modes the algorithm supports.
        /// See [`PointerModeMask`].
        PointerModeMask = CUBLASLT_ALGO_CAP_POINTER_MODE_MASK,
        /// Bitmask enumerating the kinds of postprocessing algorithm supported in the epilogue.
        /// See [`Epilogue`].
        EpilogueMask = CUBLASLT_ALGO_CAP_EPILOGUE_MASK,
        /// Array of the stage IDs the algorithm accepts for [`MatmulAlgorithmConfigAttribute::StagesId`].
        /// Query it with a zero-sized buffer to get the count; an algorithm without stage IDs reports only the undefined stage, 0.
        StagesIds = CUBLASLT_ALGO_CAP_STAGES_IDS,
        /// Nonzero when every matrix may have a negative leading dimension.
        LeadingDimensionNegative = CUBLASLT_ALGO_CAP_LD_NEGATIVE,
        /// Implementation details that affect numerical results, as [`NumericalImplFlags`].
        NumericalImplementationFlags = CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
        /// Minimum alignment required for A matrix in bytes.
        MinAlignmentABytes = CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES,
        /// Minimum alignment required for B matrix in bytes.
        MinAlignmentBBytes = CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
        /// Minimum alignment required for C matrix in bytes.
        MinAlignmentCBytes = CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES,
        /// Minimum alignment required for D matrix in bytes.
        MinAlignmentDBytes = CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES,
        /// Nonzero when the algorithm supports pointer-array batches.
        PointerArrayBatchSupport = CUBLASLT_ALGO_CAP_POINTER_ARRAY_BATCH_SUPPORT,
        /// Nonzero when the algorithm supports floating-point emulation.
        FloatingPointEmulationSupport = CUBLASLT_ALGO_CAP_FLOATING_POINT_EMULATION_SUPPORT,
        /// Experimental: nonzero when the algorithm supports [`BatchMode::Grouped`].
        PointerArrayGroupedSupport = CUBLASLT_ALGO_CAP_POINTER_ARRAY_GROUPED_SUPPORT,
    }
}

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

singe_core::native_enum! {
    /// [`MatmulAlgorithmConfigAttribute`] identifies configurable attributes of a [`MatmulAlgorithm`](crate::lt::matmul::MatmulAlgorithm).
    /// Keep algorithm-specific configuration compatible with the reported capability attributes.
    /// Use [`MatmulAlgorithm::config_attribute`](crate::lt::matmul::MatmulAlgorithm::config_attribute) and [`MatmulAlgorithm::set_config_attribute`](crate::lt::matmul::MatmulAlgorithm::set_config_attribute) to query and update them.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[non_exhaustive]
    #[repr(u32)]
    pub enum MatmulAlgorithmConfigAttribute: sys::cublasLtMatmulAlgoConfigAttributes_t {
        /// Algorithm ID, set by [`matmul_algorithm`](crate::lt::matmul::matmul_algorithm) and read-only.
        Id = CUBLASLT_ALGO_CONFIG_ID,
        /// Tile ID, one of [`MatmulAlgorithmCapAttribute::TileIds`]; the default is the undefined tile, 0.
        TileId = CUBLASLT_ALGO_CONFIG_TILE_ID,
        /// Number of K splits computed in parallel.
        /// Above 1, the parts are accumulated by [`MatmulAlgorithmConfigAttribute::ReductionScheme`].
        SplitKCount = CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
        /// [`ReductionScheme`] used when the split-K count is above 1; the default is [`ReductionScheme::None`].
        ReductionScheme = CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        /// 1 enables CTA swizzling, which changes the mapping from grid coordinates to parts of the matrices; other values than 0 and 1 are reserved.
        CtaSwizzling = CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
        /// Algorithm-specific option, up to [`MatmulAlgorithmCapAttribute::CustomOptionMax`].
        CustomOption = CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
        /// Stage ID, one of [`MatmulAlgorithmCapAttribute::StagesIds`]; the default is the undefined stage, 0.
        StagesId = CUBLASLT_ALGO_CONFIG_STAGES_ID,
        /// Inner shape ID; the default is the undefined shape, 0.
        InnerShapeId = CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
        /// Cluster shape ID; the default lets cuBLASLt choose.
        ClusterShapeId = CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
    }
}

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
    /// [`PointerModeMask`] is an enumerated type used to define and query the pointer mode capability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PointerModeMask: u32 {
        /// Supports [`PointerMode::Host`].
        const HOST_POINTER = sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_HOST.0;
        /// Supports [`PointerMode::Device`].
        const DEVICE_POINTER = sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_DEVICE.0;
        /// Supports [`PointerMode::DeviceVector`].
        const DEVICE_VECTOR_POINTER =
            sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_DEVICE_VECTOR.0;
        /// Supports [`PointerMode::AlphaDeviceVectorBetaZero`].
        const ALPHA_DEVICE_VECTOR_BETA_ZERO =
            sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO.0;
        /// Supports [`PointerMode::AlphaDeviceVectorBetaHost`].
        const ALPHA_DEVICE_VECTOR_BETA_HOST =
            sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_HOST.0;
    }
}

impl Display for PointerModeMask {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    /// Algorithm implementation details that can affect numerical results.
    ///
    /// Flags may be combined to filter multiply-accumulate instruction families and accumulator precision.
    pub struct NumericalImplFlags: u64 {
        /// Uses fused multiply-add instructions.
        const FMA = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA as _;
        /// Uses floating-point tensor operations.
        const HMMA = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA as _;
        /// Uses integer tensor operations.
        const IMMA = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_IMMA as _;
        /// Uses double-precision tensor operations.
        const DMMA = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_DMMA as _;
        /// Mask of the tensor operation flags.
        const TENSOR_OP_MASK = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK as _;
        /// Mask of the multiply-accumulate instruction flags.
        const OP_TYPE_MASK = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_TYPE_MASK as _;
        /// Accumulates the inner dot product in half precision.
        const ACCUMULATOR_16F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F as _;
        /// Accumulates the inner dot product in single precision.
        const ACCUMULATOR_32F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F as _;
        /// Accumulates the inner dot product in double precision.
        const ACCUMULATOR_64F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_64F as _;
        /// Accumulates the inner dot product in 32-bit signed integers.
        const ACCUMULATOR_32I = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I as _;
        /// Mask of the accumulator flags.
        const ACCUMULATOR_TYPE_MASK =
            sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK as _;
        /// Multiplies half-precision inputs.
        const INPUT_16F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F as _;
        /// Multiplies bfloat16 inputs.
        const INPUT_16BF = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF as _;
        /// Multiplies TF32 inputs.
        const INPUT_TF32 = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32 as _;
        /// Multiplies single-precision inputs.
        const INPUT_32F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_32F as _;
        /// Multiplies double-precision inputs.
        const INPUT_64F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_64F as _;
        /// Multiplies 8-bit integer inputs.
        const INPUT_8I = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8I as _;
        /// Multiplies 8-bit E4M3 floating-point inputs.
        const INPUT_8F_E4M3 = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E4M3 as _;
        /// Multiplies 8-bit E5M2 floating-point inputs.
        const INPUT_8F_E5M2 = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E5M2 as _;
        /// Mask of the input type flags.
        const OP_INPUT_TYPE_MASK = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK as _;
        /// Uses Gauss's complexity reduction for complex multiplication.
        const GAUSSIAN = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_GAUSSIAN as _;
    }
}

impl Display for NumericalImplFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

singe_core::native_enum! {
    /// Experimental policy used to request batch-invariant matrix multiplication results.
    ///
    /// The alignment choices are mutually exclusive, not combinable bit flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum BatchInvariance: sys::cublasLtMatmulBatchInvarianceFlags_t {
        /// Batch invariance disabled.
        Disabled = CUBLASLT_BATCH_INVARIANCE_DISABLED,
        /// Batch invariance assumes 2-byte alignment for tensors.
        Align2B = CUBLASLT_BATCH_INVARIANCE_ALIGN_2B,
        /// Batch invariance assumes 4-byte alignment for tensors.
        Align4B = CUBLASLT_BATCH_INVARIANCE_ALIGN_4B,
        /// Batch invariance assumes 8-byte alignment for tensors.
        Align8B = CUBLASLT_BATCH_INVARIANCE_ALIGN_8B,
        /// Batch invariance assumes 16-byte alignment for tensors.
        Align16B = CUBLASLT_BATCH_INVARIANCE_ALIGN_16B,
        /// The alignment class is detected from the shape, pointers and minimum alignment preference attributes of each call.
        /// Results are batch invariant only if every call detects the same class.
        Auto = CUBLASLT_BATCH_INVARIANCE_AUTO,
        /// Not set through the attribute, so the `CUBLAS_BATCH_INVARIANCE_FLAGS` environment variable decides; this is the default.
        Unset = CUBLASLT_BATCH_INVARIANCE_UNSET,
    }
}

impl_enum_display!(BatchInvariance, {
    Self::Disabled => "CUBLASLT_BATCH_INVARIANCE_DISABLED",
    Self::Align2B => "CUBLASLT_BATCH_INVARIANCE_ALIGN_2B",
    Self::Align4B => "CUBLASLT_BATCH_INVARIANCE_ALIGN_4B",
    Self::Align8B => "CUBLASLT_BATCH_INVARIANCE_ALIGN_8B",
    Self::Align16B => "CUBLASLT_BATCH_INVARIANCE_ALIGN_16B",
    Self::Auto => "CUBLASLT_BATCH_INVARIANCE_AUTO",
    Self::Unset => "CUBLASLT_BATCH_INVARIANCE_UNSET",
});
