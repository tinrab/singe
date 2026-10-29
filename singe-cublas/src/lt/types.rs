use std::fmt::{self, Display, Formatter};

use bitflags::parser;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cublas_sys as sys;

/// Matrix data ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum Order {
    /// Data is ordered in column-major format. The leading dimension is the stride (in elements) to the beginning of next column in memory.
    ColumnMajor = sys::cublasLtOrder_t::CUBLASLT_ORDER_COL as _,
    /// Data is ordered in row-major format. The leading dimension is the stride (in elements) to the beginning of next row in memory.
    RowMajor = sys::cublasLtOrder_t::CUBLASLT_ORDER_ROW as _,
    /// Data is ordered in column-major ordered tiles of 32 columns. The leading dimension is the stride (in elements) to the beginning of next group of 32-columns. For example, if the matrix has 33 columns and 2 rows, then the leading dimension must be at least `32 * 2 = 64`.
    Column32 = sys::cublasLtOrder_t::CUBLASLT_ORDER_COL32 as _,
    /// Data is ordered in column-major ordered tiles of composite tiles with total 32 columns and 8 rows. A tile is composed of interleaved inner tiles of 4 columns within 4 even or odd rows in an alternating pattern. The leading dimension is the stride (in elements) to the beginning of the first 32 column x 8 row tile for the next 32-wide group of columns. For example, if the matrix has 33 columns and 1 row, the leading dimension must be at least `(32 * 8) * 1 = 256`.
    Column4_4R2_8C = sys::cublasLtOrder_t::CUBLASLT_ORDER_COL4_4R2_8C as _,
    /// Data is ordered in column-major ordered tiles of composite tiles with total 32 columns ands 32 rows. Element offset within the tile is calculated as `(((row % 8) / 2 * 4 + row / 8) * 2 + row % 2) * 32 + col`. Leading dimension is the stride (in elements) to the beginning of the first 32 column x 32 row tile for the next 32-wide group of columns. E.g. if matrix has 33 columns and 1 row, then its leading dimensions must be at least `(32 * 32) * 1 = 1024`.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum BatchMode {
    /// The matrices of each instance of the batch are located at fixed offsets in number of elements from their locations in the previous instance.
    Strided = sys::cublasLtBatchMode_t::CUBLASLT_BATCH_MODE_STRIDED as _,
    /// The address of the matrix of each instance of the batch are read from device arrays of pointers.
    PointerArray = sys::cublasLtBatchMode_t::CUBLASLT_BATCH_MODE_POINTER_ARRAY as _,
    /// Experimental: The address of the matrix of each instance of the group are read from device arrays of pointers. Each group can have different columns, rows, and leading dimensions. See [cublasLtMatrixLayout_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatrixlayout-t) for more details.
    Grouped = sys::cublasLtBatchMode_t::CUBLASLT_BATCH_MODE_GROUPED as _,
}

impl_enum_conversion!(sys::cublasLtBatchMode_t, BatchMode);

impl_enum_display!(BatchMode, {
    BatchMode::Strided => "CUBLASLT_BATCH_MODE_STRIDED",
    BatchMode::PointerArray => "CUBLASLT_BATCH_MODE_POINTER_ARRAY",
    BatchMode::Grouped => "CUBLASLT_BATCH_MODE_GROUPED",
});

/// Experimental integer width for grouped-matrix dimension arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum IntegerWidth {
    /// 32-bit integer width
    Bits32 = sys::cublasLtIntegerWidth_t::CUBLASLT_INTEGER_WIDTH_32 as _,
    /// 64-bit integer width
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum MatrixLayoutAttribute {
    /// Specifies the data precision type. See [cudaDataType_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cudadatatype-t).
    Type = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_TYPE as _,
    /// Specifies the memory order of the data of the matrix. Default value is `CUBLASLT_ORDER_COL`. See [cublasLtOrder_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltorder-t).
    Order = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ORDER as _,
    /// Describes the number of rows in the matrix. Normally only values that can be expressed as `int32_t` are supported.
    Rows = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_ROWS as _,
    /// Describes the number of columns in the matrix. Normally only values that can be expressed as `int32_t` are supported.
    Cols = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_COLS as _,
    /// The leading dimension of the matrix. For `CUBLASLT_ORDER_COL` this is the stride (in elements) of matrix column. See also [cublasLtOrder_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltorder-t). * Currently only non-negative values are supported. * Must be large enough so that matrix memory locations are not overlapping (e.g., greater or equal to `CUBLASLT_MATRIX_LAYOUT_ROWS` in case of `CUBLASLT_ORDER_COL`).
    LeadingDimension = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_LD as _,
    /// Number of matmul operations to perform in the batch. Default value is 1. See also `CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT`, `CUBLASLT_ALGO_CAP_POINTER_ARRAY_BATCH_SUPPORT` and `CUBLASLT_ALGO_CAP_POINTER_ARRAY_GROUPED_SUPPORT` in [cublasLtMatmulAlgoCapAttributes_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulalgocapattributes-t).
    BatchCount = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT as _,
    /// Stride (in elements) to the next matrix for the strided batch operation. Default value is 0. When matrix type is planar-complex (`CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET` != 0), batch stride is interpreted by [cublasLtMatmul()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmul) in number of real valued sub-elements. E.g. for data of type `CUDA_C_16F`, offset of 1024B is encoded as a stride of value 512 (since each element of the real and imaginary matrices is a 2B (16bit) floating point type). NOTE: A bug in [cublasLtMatrixTransform()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatrixtransform) causes it to interpret the batch stride for a planar-complex matrix as if it was specified in number of complex elements. Therefore an offset of 1024B must be encoded as stride value 256 when calling [cublasLtMatrixTransform()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatrixtransform) (each complex element is 4B with real and imaginary values 2B each). This behavior is expected to be corrected in the next major cuBLAS version.
    StridedBatchOffset =
        sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET as _,
    /// Stride (in bytes) to the imaginary plane for planar-complex layout. Default value is 0, indicating that the layout is regular (real and imaginary parts of complex numbers are interleaved in memory for each element).
    PlaneOffset = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET as _,
    /// The batch mode of the matrix. Default value is `CUBLASLT_BATCH_MODE_STRIDED`. See [cublasLtBatchMode_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltbatchmode-t).
    BatchMode = sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_MODE as _,
    /// Experimental: a device pointer to the array of rows in a grouped matrix. The length of the array must be equal to `CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT`. The width of each element in the array is determined by `CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH`. Default value is NULL.
    GroupedRowsArray =
        sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_ARRAY as _,
    /// Experimental: a device pointer to the array of columns in a grouped matrix. The length of the array must be equal to `CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT`. The width of each element in the array is determined by `CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH`. Default value is NULL.
    GroupedColsArray =
        sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_GROUPED_MATRIX_LAYOUT_COLS_ARRAY as _,
    /// Experimental: a device pointer to the array of leading dimensions in a grouped matrix. The length of the array must be equal to `CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT`. The width of each element in the array is determined by `CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH`. Default value is NULL.
    GroupedLeadingDimensionArray =
        sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY as _,
    /// Experimental integer width for grouped row and column arrays.
    ///
    /// Stored as `i32`, with [`IntegerWidth::Bits32`] as the default.
    GroupedRowsColsArrayIntegerWidth = sys::cublasLtMatrixLayoutAttribute_t::
        CUBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH as _,
    /// Experimental integer width for grouped leading-dimension arrays.
    ///
    /// Stored as `i32`, with [`IntegerWidth::Bits32`] as the default.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum PointerMode {
    /// matches `CUBLAS_POINTER_MODE_HOST`, pointer targets a single value host memory
    Host = sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST as _,
    /// matches `CUBLAS_POINTER_MODE_DEVICE`, pointer targets a single value device memory
    Device = sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE as _,
    /// pointer targets an array in device memory
    DeviceVector = sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE_VECTOR as _,
    /// alpha pointer targets an array in device memory, beta is zero. Note:      `CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE` is not supported, must be 0.
    AlphaDeviceVectorBetaZero =
        sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO as _,
    /// alpha pointer targets an array in device memory, beta is a single value in host memory.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum MatmulDescriptorAttribute {
    /// Requests batch-invariant matrix multiplication results.
    BatchInvarianceFlags =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BATCH_INVARIANCE_FLAGS as _,
    /// Compute type. Defines the data type used for multiply and accumulate operations, and the accumulator during the matrix multiplication. See [cublasComputeType_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublascomputetype-t).
    ComputeType = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_COMPUTE_TYPE as _,
    /// Scale type. Defines the data type of the scaling factors `alpha` and `beta`. The accumulator value and the value from matrix `C` are typically converted to scale type before final scaling. The value is then converted from scale type to the type of matrix `D` before storing in memory. The default value depends on `CUBLASLT_MATMUL_DESC_COMPUTE_TYPE`. See [cudaDataType_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cudadatatype-t).
    ScaleType = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_SCALE_TYPE as _,
    /// Specifies `alpha` and `beta` are passed by reference, whether they are scalars on the host or on the device, or device vectors. Default value is: `CUBLASLT_POINTER_MODE_HOST` (i.e., on the host). See [cublasLtPointerMode_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltpointermode-t).
    PointerMode = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_POINTER_MODE as _,
    /// Specifies the type of transformation operation that should be performed on matrix A. Default value is: `CUBLAS_OP_N` (i.e., non-transpose operation). See [cublasOperation_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasoperation-t).
    TransposeA = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA as _,
    /// Specifies the type of transformation operation that should be performed on matrix B. Default value is: `CUBLAS_OP_N` (i.e., non-transpose operation). See [cublasOperation_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasoperation-t).
    TransposeB = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB as _,
    /// Specifies the type of transformation operation that should be performed on matrix C. Currently only `CUBLAS_OP_N` is supported. Default value is: `CUBLAS_OP_N` (i.e., non-transpose operation). See [cublasOperation_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasoperation-t).
    TransposeC = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSC as _,
    /// Indicates whether the lower or upper part of the dense matrix was filled, and consequently should be used by the function. Currently this flag is not supported for bfloat16 or FP8 data types and is not supported on the following GPUs: Hopper, Blackwell. Default value is: `CUBLAS_FILL_MODE_FULL`. See [cublasFillMode_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasfillmode-t).
    FillMode = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FILL_MODE as _,
    /// Epilogue function. See [cublasLtEpilogue_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltepilogue-t). Default value is: `CUBLASLT_EPILOGUE_DEFAULT`.
    Epilogue = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE as _,
    /// Bias or Bias gradient vector pointer in the device memory. * Input vector with length that matches the number of rows of matrix D when one of the following epilogues is used: `CUBLASLT_EPILOGUE_BIAS`, `CUBLASLT_EPILOGUE_RELU_BIAS`, `CUBLASLT_EPILOGUE_RELU_AUX_BIAS`, `CUBLASLT_EPILOGUE_GELU_BIAS`, `CUBLASLT_EPILOGUE_GELU_AUX_BIAS`. * Output vector with length that matches the number of rows of matrix D when one of the following epilogues is used: `CUBLASLT_EPILOGUE_DRELU_BGRAD`, `CUBLASLT_EPILOGUE_DGELU_BGRAD`, `CUBLASLT_EPILOGUE_BGRADA`. * Output vector with length that matches the number of columns of matrix D when one of the following epilogues is used: `CUBLASLT_EPILOGUE_BGRADB`. Bias vector elements are the same type as `alpha` and `beta` (see `CUBLASLT_MATMUL_DESC_SCALE_TYPE` in this table) when matrix D datatype is `CUDA_R_8I` and same as matrix D datatype otherwise. See the datatypes table under [cublasLtMatmul()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmul) for detailed mapping. Default value is: NULL.
    BiasPointer = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER as _,
    /// Stride (in elements) to the next bias or bias gradient vector for strided batch operations. If `cublasLtBatchMode_t` of any matrix is set to `CUBLASLT_BATCH_MODE_GROUPED` and `CUBLASLT_MATMUL_DESC_EPILOGUE` includes `CUBLASLT_EPILOGUE_BIAS` then `CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE` must be set to 1. The default value is 0.
    BiasBatchStride =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE as _,
    /// Pointer for epilogue auxiliary buffer. * Output vector for `ReLu` bit-mask in forward pass when `CUBLASLT_EPILOGUE_RELU_AUX` or `CUBLASLT_EPILOGUE_RELU_AUX_BIAS` epilogue is used. * Input vector for `ReLu` bit-mask in backward pass when `CUBLASLT_EPILOGUE_DRELU` or `CUBLASLT_EPILOGUE_DRELU_BGRAD` epilogue is used. * Output of GELU input matrix in forward pass when `CUBLASLT_EPILOGUE_GELU_AUX_BIAS` epilogue is used. * Input of GELU input matrix for backward pass when `CUBLASLT_EPILOGUE_DGELU` or `CUBLASLT_EPILOGUE_DGELU_BGRAD` epilogue is used. For aux data type, see `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE`. Routines that don't dereference this pointer, like [cublasLtMatmulAlgoGetHeuristic()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulalgogetheuristic) depend on its value to determine expected pointer alignment. Requires setting the `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD` attribute.
    EpilogueAuxPointer =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER as _,
    /// Leading dimension for epilogue auxiliary buffer. * `ReLu` bit-mask matrix leading dimension in elements (i.e. bits) when `CUBLASLT_EPILOGUE_RELU_AUX`, `CUBLASLT_EPILOGUE_RELU_AUX_BIAS`, `CUBLASLT_EPILOGUE_DRELU_BGRAD`, or `CUBLASLT_EPILOGUE_DRELU_BGRAD` epilogue is used. Must be divisible by 128 and be no less than the number of rows in the output matrix. * GELU input matrix leading dimension in elements when `CUBLASLT_EPILOGUE_GELU_AUX_BIAS`, `CUBLASLT_EPILOGUE_DGELU`, or `CUBLASLT_EPILOGUE_DGELU_BGRAD` epilogue used. Must be divisible by 8 and be no less than the number of rows in the output matrix.
    EpilogueAuxLeadingDimension =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD as _,
    /// Batch stride for epilogue auxiliary buffer. * `ReLu` bit-mask matrix batch stride in elements (i.e. bits) when `CUBLASLT_EPILOGUE_RELU_AUX`, `CUBLASLT_EPILOGUE_RELU_AUX_BIAS` or `CUBLASLT_EPILOGUE_DRELU_BGRAD` epilogue is used. Must be divisible by 128. * GELU input matrix batch stride in elements when `CUBLASLT_EPILOGUE_GELU_AUX_BIAS`, `CUBLASLT_EPILOGUE_DRELU`, or `CUBLASLT_EPILOGUE_DGELU_BGRAD` epilogue used. Must be divisible by 8. Default value: 0.
    EpilogueAuxBatchStride =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE as _,
    /// Batch stride for alpha vector. Used together with `CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST` when matrix D's `CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT` is greater than 1. If `CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO` is set then `CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE` must be set to 0 as this mode doesn't support batched alpha vector. If `cublasLtBatchMode_t` of any matrix is not set to `CUBLASLT_BATCH_MODE_STRIDED` then `CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE` must be set to 0. Default value: 0.
    AlphaVectorBatchStride =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE as _,
    /// Number of SMs to target for parallel execution. Optimizes heuristics for execution on a different number of SMs when user expects a concurrent stream to be using some of the device resources. Default value: 0.
    SmCountTarget = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET as _,
    /// Device pointer to the scale factor value that converts data in matrix A to the compute data type range. The scaling factor must have the same type as the compute type. If not specified, or set to NULL, the scaling factor is assumed to be 1. If set for an unsupported matrix data, scale, and compute type combination, calling [cublasLtMatmul()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmul) will return `CUBLAS_INVALID_VALUE`. Default value: NULL
    AScalePointer = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER as _,
    /// Equivalent to `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` for matrix B. Default value: NULL
    BScalePointer = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER as _,
    /// Equivalent to `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` for matrix C. Default value: NULL
    CScalePointer = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_C_SCALE_POINTER as _,
    /// Equivalent to `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` for matrix D. Default value: NULL
    DScalePointer = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_D_SCALE_POINTER as _,
    /// Device pointer to the memory location that on completion will be set to the maximum of absolute values in the output matrix. The computed value has the same type as the compute type. If not specified, or set to NULL, the maximum absolute value is not computed. If set for an unsupported matrix data, scale, and compute type combination, calling [cublasLtMatmul()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmul) will return `CUBLAS_INVALID_VALUE`. Default value: NULL
    AmaxDPointer = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_AMAX_D_POINTER as _,
    /// The type of the data that will be stored in `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`. If unset (or set to the default value of -1), the data type is set to be the output matrix element data type (`DType`) with some exceptions: * `ReLu` uses a bit-mask. * For FP8 kernels with an output type (`DType`) of `CUDA_R_8F_E4M3`, the data type can be set to a non-default value if: 1. `AType` and `BType` are `CUDA_R_8F_E4M3`. 2. Bias Type is `CUDA_R_16F`. 3. `CType` is `CUDA_R_16BF` or `CUDA_R_16F` 4. `CUBLASLT_MATMUL_DESC_EPILOGUE` is set to `CUBLASLT_EPILOGUE_GELU_AUX` When `CType` is `CUDA_R_16F`, the data type may be set to `CUDA_R_16F` or `CUDA_R_8F_E4M3`. When `CType` is `CUDA_R_16BF`, the data type may be set to `CUDA_R_16BF`. Otherwise, the data type should be left unset or set to the default value of -1. If set for an unsupported matrix data, scale, and compute type combination, calling [cublasLtMatmul()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmul) will return `CUBLAS_INVALID_VALUE`. Default value: -1
    EpilogueAuxDataType =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE as _,
    /// Device pointer to the scaling factor value to convert results from compute type data range to storage data range in the auxiliary matrix that is set via `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`. The scaling factor value must have the same type as the compute type. If not specified, or set to NULL, the scaling factor is assumed to be 1. If set for an unsupported matrix data, scale, and compute type combination, calling [cublasLtMatmul()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmul) will return `CUBLAS_INVALID_VALUE`. Default value: NULL
    EpilogueAuxScalePointer =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER as _,
    /// Device pointer to the memory location that on completion will be set to the maximum of absolute values in the buffer that is set via `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`. The computed value has the same type as the compute type. If not specified, or set to NULL, the maximum absolute value is not computed. If set for an unsupported matrix data, scale, and compute type combination, calling [cublasLtMatmul()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmul) will return `CUBLAS_INVALID_VALUE`. Default value: NULL
    EpilogueAuxAmaxPointer =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER as _,
    /// Flag for managing FP8 fast accumulation mode. When enabled, on some GPUs problem execution might be faster but at the cost of lower accuracy because intermediate results will not periodically be promoted to a higher precision. Currently this flag has an effect on the following GPUs: Ada, Hopper. Default value: 0 - fast accumulation mode is disabled
    FastAccum = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_FAST_ACCUM as _,
    /// Type of the bias or bias gradient vector in the device memory. Bias case: see `CUBLASLT_EPILOGUE_BIAS`. If unset (or set to the default value of -1), the bias vector elements are the same type as the elements of the output matrix (Dtype) with the following exceptions: * IMMA kernels with computeType=`CUDA_R_32I` and `Ctype=CUDA_R_8I` where the bias vector elements are the same type as alpha, beta (`CUBLASLT_MATMUL_DESC_SCALE_TYPE=CUDA_R_32F`) * For FP8 kernels with an output type of `CUDA_R_32F`, `CUDA_R_8F_E4M3` or `CUDA_R_8F_E5M2`. See [cublasLtMatmul()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmul) for more details. Default value: -1
    BiasDataType = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE as _,
    /// Scaling mode that defines how the matrix scaling factor for matrix A is interpreted. Default value: 0. See [cublasLtMatmulMatrixScale_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulmatrixscale-t).
    AScaleMode = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE as _,
    /// Scaling mode that defines how the matrix scaling factor for matrix B is interpreted. Default value: 0. See [cublasLtMatmulMatrixScale_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulmatrixscale-t).
    BScaleMode = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE as _,
    /// Scaling mode that defines how the matrix scaling factor for matrix C is interpreted. Default value: 0. See [cublasLtMatmulMatrixScale_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulmatrixscale-t).
    CScaleMode = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_C_SCALE_MODE as _,
    /// Scaling mode that defines how the matrix scaling factor for matrix D is interpreted. Default value: 0. See [cublasLtMatmulMatrixScale_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulmatrixscale-t).
    DScaleMode = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_D_SCALE_MODE as _,
    /// Scaling mode that defines how the matrix scaling factor for the auxiliary matrix is interpreted. Default value: 0. See [cublasLtMatmulMatrixScale_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulmatrixscale-t).
    EpilogueAuxScaleMode =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE as _,
    /// Device pointer to the scale factors that are used to convert data in matrix D to the compute data type range. The scaling factor value type is defined by the scaling mode (see `CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE`). If set for an unsupported matrix data, scale, scale mode, and compute type combination, or missing for a supported combination, then calling [cublasLtMatmul()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmul) will return `CUBLAS_INVALID_VALUE`. Default value: NULL.
    DOutScalePointer =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER as _,
    /// Scaling mode that defines how the output matrix scaling factor for matrix D is interpreted. Default value: 0. See [cublasLtMatmulMatrixScale_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulmatrixscale-t).
    DOutScaleMode = sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE as _,
    /// Emulation descriptor to configure floating point emulation parameters. Default value: NULL.
    EmulationDescriptor =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EMULATION_DESCRIPTOR as _,
    /// Experimental: Batch stride for alpha. Applicable when matrix D's `CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT` is greater than 1. Supported values are 0 and 1. Default value is 0. When the value is set to 1, the parameter `alpha` of [cublasLtMatmul()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmul) must contain a device array of pointers of length `CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT`. This setting is currently only supported if `cublasLtBatchMode_t` of all matrices is set to `CUBLASLT_BATCH_MODE_GROUPED` and `CUBLASLT_MATMUL_DESC_POINTER_MODE` is set to `CUBLASLT_POINTER_MODE_DEVICE`.
    AlphaBatchStride =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE as _,
    /// Experimental: Batch stride for beta. Applicable when matrix D's `CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT` is greater than 1. Supported values are 0 and 1. Default value is 0. When the value is set to 1, the parameter `beta` of [cublasLtMatmul()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmul) must contain a device array of pointers of length `CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT`. This setting is currently only supported if `cublasLtBatchMode_t` of all matrices is set to `CUBLASLT_BATCH_MODE_GROUPED` and `CUBLASLT_MATMUL_DESC_POINTER_MODE` is set to `CUBLASLT_POINTER_MODE_DEVICE`.
    BetaBatchStride =
        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE as _,
}

impl_enum_conversion!(
    sys::cublasLtMatmulDescAttributes_t,
    MatmulDescriptorAttribute,
);

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

/// Scaling mode that defines how scaling factor pointers are interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum MatrixScale {
    /// Scaling factors are tensors of packed `u8` (UE8M0) values in a simple column-major layout.
    /// Four E8M0 exponents covering consecutive 128-element K-blocks are packed into one `u32`.
    /// The `u32` elements are stored M-major (for A) or N-major (for B) with shape
    /// `round_up(M,4)` x ceil(K/512) (A) or `round_up(N,4)` x ceil(K/512) (B).
    Vector128MnK4UE8M0 =
        sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_MN_K4_UE8M0 as _,
    /// Same as [`Self::Vector128MnK4UE8M0`] but with 32-element K-block granularity.
    /// Four E8M0 exponents covering consecutive 32-element K-blocks are packed into one `u32`.
    /// Shape is `round_up(M,4)` x ceil(K/128) (A) or `round_up(N,4)` x ceil(K/128) (B).
    Vector32MnK4UE8M0 =
        sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_MN_K4_UE8M0 as _,
    /// Scaling factors are single precision scalars applied to the whole tensor.
    Scalar32F = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F as _,
    /// Scaling factors are vectors of `f32` values. This mode is only applicable to matrices A and B, in which case
    /// the vectors are expected to have M and N elements respectively, and each (i, j)-th element of product of A and B is
    /// multiplied by i-th element of A scale and j-th element of B scale.
    OuterVector32F =
        sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F as _,
    /// Same as [`Self::Vector16UE4M3`], except that scaling factor tensor elements have type UE8M0 and the block size is 32 elements.
    Vector32UE8M0 = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 as _,
    /// Scaling factors are tensors that contain a dedicated scaling factor stored as an 8-bit UE4M3 value for
    /// each 16-element block in the innermost dimension of the corresponding data tensor.
    Vector16UE4M3 = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 as _,
    /// Scaling factors are tensors that contain a dedicated `f32` scaling factor for each 128-element block in the
    /// innermost dimension of the corresponding data tensor.
    Vector12832F = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F as _,
    /// Scaling factors are tensors that contain a dedicated `f32` scaling factor for each 128x128-element block in
    /// the the corresponding data tensor.
    Block128x12832F =
        sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F as _,
    /// Experimental: Scaling factors are single precision scalars applied to the whole tensor. Each batch has its own
    /// scaling factor.
    PerBatchScalar32F =
        sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_PER_BATCH_SCALAR_32F as _,
}

impl_enum_conversion!(sys::cublasLtMatmulMatrixScale_t, MatrixScale);

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

/// [`MatrixTransformDescAttribute`] identifies the configurable attributes of a
/// [`MatrixTransformDescriptor`](crate::lt::descriptor::MatrixTransformDescriptor).
/// Use
/// [`MatrixTransformDescriptor::attribute`](crate::lt::descriptor::MatrixTransformDescriptor::attribute)
/// and
/// [`MatrixTransformDescriptor::set_attribute`](crate::lt::descriptor::MatrixTransformDescriptor::set_attribute)
/// to query and update them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum MatrixTransformDescAttribute {
    /// Scale type. Inputs are converted to the scale type for scaling and summation, and results are then converted to the output type to store in the memory. For the supported data types see [cudaDataType_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cudadatatype-t).
    ScaleType =
        sys::cublasLtMatrixTransformDescAttributes_t::CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE
            as _,
    /// Specifies the scalars alpha and beta are passed by reference whether on the host or on the device. Default value is: `CUBLASLT_POINTER_MODE_HOST` (i.e., on the host). See [cublasLtPointerMode_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltpointermode-t).
    PointerMode =
        sys::cublasLtMatrixTransformDescAttributes_t::CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE
            as _,
    /// Specifies the type of operation that should be performed on the matrix A. Default value is: `CUBLAS_OP_N` (i.e., non-transpose operation). See [cublasOperation_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasoperation-t).
    TransposeA =
        sys::cublasLtMatrixTransformDescAttributes_t::CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA as _,
    /// Specifies the type of operation that should be performed on the matrix B. Default value is: `CUBLAS_OP_N` (i.e., non-transpose operation). See [cublasOperation_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasoperation-t).
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum EmulationDescAttribute {
    /// Strategy, see [cublasEmulationStrategy_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasemulationstrategy-t). Defines when to use floating point emulation algorithms. Default: `EMULATION_STRATEGY_DEFAULT`.
    Strategy = sys::cublasLtEmulationDescAttributes_t::CUBLASLT_EMULATION_DESC_STRATEGY as _,
    /// Special values support, see [cudaEmulationSpecialValuesSupport_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cudaemulationspecialvaluessupport-t). Defines a bit mask of special cases in floating-point representations that must be supported. Default: `EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT`.
    SpecialValuesSupport =
        sys::cublasLtEmulationDescAttributes_t::CUBLASLT_EMULATION_DESC_SPECIAL_VALUES_SUPPORT
            as _,
    /// Selects dynamic or fixed mantissa control for fixed-point emulation.
    ///
    /// Stored as `i32`, using the values of [`EmulationMantissaControl`](singe_cuda::types::EmulationMantissaControl); the default value is zero.
    /// Dynamic mode uses the recommended bit count plus the configured offset, subject to quantization and the maximum bit count.
    /// Dynamic mode falls back to floating-point computation when the quantized requirement exceeds the quantized maximum.
    /// Fixed mode uses the quantized maximum bit count.
    FixedPointMantissaControl = sys::cublasLtEmulationDescAttributes_t::
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_CONTROL as _,
    /// Maximum retained mantissa bit count for fixed-point emulation, subject to quantization.
    ///
    /// Stored as `i32`; the default of zero lets the library choose a device-dependent limit.
    FixedPointMaxMantissaBitCount = sys::cublasLtEmulationDescAttributes_t::
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MAX_MANTISSA_BIT_COUNT as _,
    /// Bias added to the recommended mantissa bit count in dynamic mode.
    ///
    /// Stored as `i32`, with a default of zero.
    FixedPointMantissaBitOffset = sys::cublasLtEmulationDescAttributes_t::
        CUBLASLT_EMULATION_DESC_FIXEDPOINT_MANTISSA_BIT_OFFSET as _,
    /// Device pointer to an `i32` receiving the actual retained mantissa bit count.
    ///
    /// The library writes `-1` when fixed-point emulation is not used.
    /// The default pointer is null.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum ReductionScheme {
    /// No reduction scheme, dot-product shall be performed in one sequence.
    None = sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_NONE as _,
    /// Reduction is performed "in place" using the output buffer, parts are added up in the output data type. Workspace is only used for counters that guarantee sequentiality.
    InPlace = sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_INPLACE as _,
    /// Intermediate results are stored in compute type in the workspace and reduced in a separate step.
    ComputeType = sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE as _,
    /// Intermediate results are stored in output type in the workspace and reduced in a separate step.
    OutputType = sys::cublasLtReductionScheme_t::CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE as _,
    /// Intermediate results are stored in output type in the workspace and reduced in a separate step.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum Epilogue {
    /// No special postprocessing, just scale and quantize results if necessary.
    Default = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT as _,
    /// `ReLu`, apply `ReLu` point-wise transform to the results (x:=max(x, 0)).
    Relu = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU as _,
    /// `ReLu`, apply `ReLu` point-wise transform to the results (x:=max(x, 0)).
    ///
    /// This epilogue mode produces an extra output, a `ReLu` bit-mask matrix,  see `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`.
    ReluAux = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_AUX as _,
    /// Bias, apply (broadcasted) Bias from bias vector. Bias vector length must match matrix D rows, it must be packed  (stride between vector elements is 1). Bias vector is broadcasted to all columns and added before applying final  postprocessing.
    Bias = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS as _,
    /// `ReLu` and Bias, apply Bias and then `ReLu` transform
    ReluBias = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS as _,
    /// `ReLu` and Bias, apply Bias and then `ReLu` transform
    ///
    /// This epilogue mode produces an extra output, a `ReLu` bit-mask matrix,  see `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`.
    ReluAuxBias = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_AUX_BIAS as _,
    /// `ReLu` gradient. Apply `ReLu` gradient to matmul output. Store `ReLu` gradient in the output matrix.
    ///
    /// This epilogue mode requires an extra input,  see `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`.
    DRelu = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DRELU as _,
    /// `ReLu` and Bias gradients. Apply independently `ReLu` and Bias gradient to  matmul output. Store `ReLu` gradient in the output matrix, and Bias gradient  in the auxiliary output (see `CUBLASLT_MATMUL_DESC_BIAS_POINTER`).
    ///
    /// This epilogue mode requires an extra input,  see `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`.
    DReluBGrad = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DRELU_BGRAD as _,
    /// GELU, apply GELU point-wise transform to the results (x:=GELU(x)).
    Gelu = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU as _,
    /// GELU, apply GELU point-wise transform to the results (x:=GELU(x)).
    ///
    /// This epilogue mode outputs GELU input as a separate matrix (useful for training).  See `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`.
    GeluAux = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_AUX as _,
    /// GELU and Bias, apply Bias and then GELU transform
    GeluBias = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS as _,
    /// GELU and Bias, apply Bias and then GELU transform
    ///
    /// This epilogue mode outputs GELU input as a separate matrix (useful for training).  See `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`.
    GeluAuxBias = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_AUX_BIAS as _,
    /// GELU gradient. Apply GELU gradient to matmul output. Store GELU gradient in the output matrix.
    ///
    /// This epilogue mode requires an extra input,  see `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`.
    DGelu = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DGELU as _,
    /// GELU and Bias gradients. Apply independently GELU and Bias gradient to  matmul output. Store GELU gradient in the output matrix, and Bias gradient  in the auxiliary output (see `CUBLASLT_MATMUL_DESC_BIAS_POINTER`).
    ///
    /// This epilogue mode requires an extra input,  see `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`.
    DGeluBGrad = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DGELU_BGRAD as _,
    /// Bias gradient based on the input matrix A.
    ///
    /// The bias size corresponds to the number of rows of the matrix D.  The reduction happens over the GEMM's "k" dimension.
    ///
    /// Stores Bias gradient in the auxiliary output  (see `CUBLASLT_MATMUL_DESC_BIAS_POINTER`).
    BGradA = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BGRADA as _,
    /// Bias gradient based on the input matrix B.
    ///
    /// The bias size corresponds to the number of columns of the matrix D.  The reduction happens over the GEMM's "k" dimension.
    ///
    /// Stores Bias gradient in the auxiliary output  (see `CUBLASLT_MATMUL_DESC_BIAS_POINTER`).
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum SearchMode {
    /// ask heuristics for best algo for given usecase
    BestFit = sys::cublasLtMatmulSearch_t::CUBLASLT_SEARCH_BEST_FIT as _,
    /// only try to find best config for preconfigured algo id
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum MatmulPreferenceAttribute {
    /// Search mode. See [cublasLtMatmulSearch_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulsearch-t). Default is `CUBLASLT_SEARCH_BEST_FIT`.
    SearchMode = sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_SEARCH_MODE as _,
    /// Maximum allowed workspace memory. Default is 0 (no workspace memory allowed).
    MaxWorkspaceBytes =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES as _,
    /// Reduction scheme mask. See [cublasLtReductionScheme_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltreductionscheme-t). Only algorithm configurations specifying `CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME` that is not masked out by this attribute are allowed. For example, a mask value of 0x03 will allow only `INPLACE` and `COMPUTE_TYPE` reduction schemes. Default is `CUBLASLT_REDUCTION_SCHEME_MASK` (i.e., allows all reduction schemes).
    ReductionSchemeMask =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK as _,
    /// Minimum buffer alignment for matrix A (in bytes). Selecting a smaller value will exclude algorithms that can not work with matrix A, which is not as strictly aligned as the algorithms need. Default is 256 bytes.
    MinAlignmentABytes =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES as _,
    /// Minimum buffer alignment for matrix B (in bytes). Selecting a smaller value will exclude algorithms that can not work with matrix B, which is not as strictly aligned as the algorithms need. Default is 256 bytes.
    MinAlignmentBBytes =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES as _,
    /// Minimum buffer alignment for matrix C (in bytes). Selecting a smaller value will exclude algorithms that can not work with matrix C, which is not as strictly aligned as the algorithms need. Default is 256 bytes.
    MinAlignmentCBytes =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES as _,
    /// Minimum buffer alignment for matrix D (in bytes). Selecting a smaller value will exclude algorithms that can not work with matrix D, which is not as strictly aligned as the algorithms need. Default is 256 bytes.
    MinAlignmentDBytes =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES as _,
    /// Maximum wave count. See [cublasLtMatmulHeuristicResult_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulheuristicresult-t)`::wavesCount.` Selecting a non-zero value will exclude algorithms that report device utilization higher than specified. Default is `0.0f.`
    MaxWavesCount =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT as _,
    /// Numerical implementation details mask. See [cublasLtNumericalImplFlags_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltnumericalimplflags-t). Filters heuristic result to only include algorithms that use the allowed implementations. default: uint64_t(-1) (allow everything)
    ImplementationMask =
        sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_IMPL_MASK as _,
    /// Experimental average reduction dimension for grouped matrix multiplication.
    ///
    /// Stored as `u32`, with a default of zero.
    GroupedAverageReductionDim = sys::cublasLtMatmulPreferenceAttributes_t::
        CUBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM as _,
    /// Experimental average number of rows in grouped output matrices `D`.
    ///
    /// Stored as `u32`, with a default of zero.
    GroupedDescDAverageRows = sys::cublasLtMatmulPreferenceAttributes_t::
        CUBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS as _,
    /// Experimental average number of columns in grouped output matrices `D`.
    ///
    /// Stored as `u32`, with a default of zero.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum MatmulAlgorithmCapAttribute {
    /// Support for split-K. Boolean (0 or 1) to express if split-K implementation is supported. 0 means no support, and supported otherwise. See `CUBLASLT_ALGO_CONFIG_SPLITK_NUM` of [cublasLtMatmulAlgoConfigAttributes_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulalgoconfigattributes-t).
    SplitKSupport = sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_SPLITK_SUPPORT as _,
    /// Mask to express the types of reduction schemes supported, see [cublasLtReductionScheme_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltreductionscheme-t). If the reduction scheme is not masked out then it is supported. For example: `int isReductionSchemeComputeTypeSupported ? (reductionSchemeMask & CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) == CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE ? 1 : 0;`
    ReductionSchemeMask =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK as _,
    /// Support for CTA-swizzling. Boolean (0 or 1) to express if CTA-swizzling implementation is supported. 0 means no support, and 1 means supported value of 1; other values are reserved. See also `CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING` of [cublasLtMatmulAlgoConfigAttributes_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulalgoconfigattributes-t).
    CtaSwizzlingSupport =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT as _,
    /// Support strided batch. 0 means no support, supported otherwise.
    StridedBatchSupport =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT as _,
    /// Support results out of place (D != C in D = alpha.A.B + beta.C). 0 means no support, supported otherwise.
    OutOfPlaceResultSupport =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT as _,
    /// Syrk (symmetric rank k update)/herk (Hermitian rank k update) support (on top of regular gemm). 0 means no support, supported otherwise.
    UploSupport = sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_UPLO_SUPPORT as _,
    /// The tile ids possible to use. See [cublasLtMatmulTile_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmultile-t). If no tile ids are supported then use `CUBLASLT_MATMUL_TILE_UNDEFINED`. Use [cublasLtMatmulAlgoCapGetAttribute()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulalgocapgetattribute) with `sizeInBytes = 0` to query the actual count.
    TileIds = sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_TILE_IDS as _,
    /// Custom option range is from 0 to `CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX` (inclusive). See `CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION` of [cublasLtMatmulAlgoConfigAttributes_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulalgoconfigattributes-t).
    CustomOptionMax =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX as _,
    /// Indicates whether the algorithm supports custom (not COL or ROW memory order). 0 means only COL and ROW memory order is allowed, non-zero means that algo might have different requirements. See [cublasLtOrder_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltorder-t).
    CustomMemoryOrder =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_CUSTOM_MEMORY_ORDER as _,
    /// Bitmask enumerating the pointer modes the algorithm supports. See [cublasLtPointerModeMask_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltpointermodemask-t).
    PointerModeMask =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_POINTER_MODE_MASK as _,
    /// Bitmask enumerating the kinds of postprocessing algorithm supported in the epilogue. See [cublasLtEpilogue_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltepilogue-t).
    EpilogueMask = sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_EPILOGUE_MASK as _,
    /// The stages ids possible to use. See [cublasLtMatmulStages_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulstages-t). If no stages ids are supported then use `CUBLASLT_MATMUL_STAGES_UNDEFINED`. Use [cublasLtMatmulAlgoCapGetAttribute()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulalgocapgetattribute) with `sizeInBytes = 0` to query the actual count.
    StagesIds = sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_STAGES_IDS as _,
    /// Support for negative leading dimension for all of the matrices. 0 means no support, supported otherwise.
    LeadingDimensionNegative =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_LD_NEGATIVE as _,
    /// Details about algorithm's implementation that affect it's numerical behavior. See [cublasLtNumericalImplFlags_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltnumericalimplflags-t).
    NumericalImplementationFlags =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS as _,
    /// Minimum alignment required for A matrix in bytes.
    MinAlignmentABytes =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES as _,
    /// Minimum alignment required for B matrix in bytes.
    MinAlignmentBBytes =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES as _,
    /// Minimum alignment required for C matrix in bytes.
    MinAlignmentCBytes =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES as _,
    /// Minimum alignment required for D matrix in bytes.
    MinAlignmentDBytes =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES as _,
    /// Support pointer array batch. 0 means no support, supported otherwise.
    PointerArrayBatchSupport =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_POINTER_ARRAY_BATCH_SUPPORT as _,
    /// Support for for floating point emulation. See [Floating Point Emulation](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#floating-point-emulation).
    FloatingPointEmulationSupport =
        sys::cublasLtMatmulAlgoCapAttributes_t::CUBLASLT_ALGO_CAP_FLOATING_POINT_EMULATION_SUPPORT
            as _,
    /// Experimental: Support pointer array grouped batch. 0 means no support, supported otherwise. See `CUBLASLT_BATCH_MODE_GROUPED` of [cublasLtBatchMode_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltbatchmode-t).
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[non_exhaustive]
#[repr(u32)]
pub enum MatmulAlgorithmConfigAttribute {
    /// Read-only attribute. Algorithm index. See [cublasLtMatmulAlgoGetIds()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulalgogetids). Set by [cublasLtMatmulAlgoInit()](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulalgoinit).
    Id = sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_ID as _,
    /// Tile id. See [cublasLtMatmulTile_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmultile-t). Default: `CUBLASLT_MATMUL_TILE_UNDEFINED`.
    TileId = sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_TILE_ID as _,
    /// Number of K splits. If the number of K splits is greater than one, `SPLITK_NUM` parts of matrix multiplication will be computed in parallel. The results will be accumulated according to `CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME`.
    SplitKCount = sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM as _,
    /// Reduction scheme to use when splitK value > 1. Default: `CUBLASLT_REDUCTION_SCHEME_NONE`. See [cublasLtReductionScheme_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltreductionscheme-t).
    ReductionScheme =
        sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME as _,
    /// Enable/Disable CTA swizzling. Change mapping from CUDA grid coordinates to parts of the matrices. Possible values: 0 and 1; other values reserved.
    CtaSwizzling =
        sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING as _,
    /// Custom option value. Each algorithm can support some custom options that don't fit the description of the other configuration attributes. See the `CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX` of [cublasLtMatmulAlgoCapAttributes_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulalgocapattributes-t) for the accepted range for a specific case.
    CustomOption =
        sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION as _,
    /// stages id, see [cublasLtMatmulStages_t](https://docs.nvidia.com/cuda/archive/13.4.2/cublas/index.html#cublasltmatmulstages-t). Default: `CUBLASLT_MATMUL_STAGES_UNDEFINED`.
    StagesId = sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_STAGES_ID as _,
    /// Inner shape ID. Refer to `cublasLtMatmulInnerShape_t.` Default: `CUBLASLT_MATMUL_INNER_SHAPE_UNDEFINED`.
    InnerShapeId =
        sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID as _,
    /// Cluster shape ID. Refer to `cublasLtClusterShape_t.` Default: `CUBLASLT_CLUSTER_SHAPE_AUTO`.
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
    /// `cublasLtPointerModeMask_t` is an enumerated type used to define and query the pointer mode capability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PointerModeMask: u32 {
        /// see CUBLASLT_POINTER_MODE_HOST
        const HOST_POINTER = sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_HOST as _;
        /// see CUBLASLT_POINTER_MODE_DEVICE
        const DEVICE_POINTER = sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_DEVICE as _;
        /// see CUBLASLT_POINTER_MODE_DEVICE_VECTOR
        const DEVICE_VECTOR_POINTER =
            sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_DEVICE_VECTOR as _;
        /// see CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO
        const ALPHA_DEVICE_VECTOR_BETA_ZERO =
            sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_ZERO as _;
        /// see CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST
        const ALPHA_DEVICE_VECTOR_BETA_HOST =
            sys::cublasLtPointerModeMask_t::CUBLASLT_POINTER_MODE_MASK_ALPHA_DEVICE_VECTOR_BETA_HOST as _;
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
    /// Flags may be combined to filter multiply-accumulate instruction families
    /// and accumulator precision.
    pub struct NumericalImplFlags: u64 {
        /// Specify that the implementation is based on \[H,F,D\]FMA (fused multiply-add) family instructions.
        const FMA = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_FMA as _;
        /// Specify that the implementation is based on HMMA (tensor operation) family instructions.
        const HMMA = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_HMMA as _;
        /// Specify that the implementation is based on IMMA (integer tensor operation) family instructions.
        const IMMA = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_IMMA as _;
        /// Specify that the implementation is based on DMMA (double precision tensor operation) family instructions.
        const DMMA = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_DMMA as _;
        /// Mask to filter implementations using any of the above kinds of tensor operations.
        const TENSOR_OP_MASK = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK as _;
        /// Mask to filter implementation details about multiply-accumulate instructions used.
        const OP_TYPE_MASK = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_TYPE_MASK as _;
        /// Specify that the implementation's inner dot product is using half precision accumulator.
        const ACCUMULATOR_16F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_16F as _;
        /// Specify that the implementation's inner dot product is using single precision accumulator.
        const ACCUMULATOR_32F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F as _;
        /// Specify that the implementation's inner dot product is using double precision accumulator.
        const ACCUMULATOR_64F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_64F as _;
        /// Specify that the implementation's inner dot product is using 32 bit signed integer precision accumulator.
        const ACCUMULATOR_32I = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32I as _;
        /// Mask to filter implementation details about accumulator used.
        const ACCUMULATOR_TYPE_MASK =
            sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK as _;
        /// Specify that the implementation's inner dot product multiply-accumulate instruction is using half-precision inputs.
        const INPUT_16F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16F as _;
        /// Specify that the implementation's inner dot product multiply-accumulate instruction is using bfloat16 inputs.
        const INPUT_16BF = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_16BF as _;
        /// Specify that the implementation's inner dot product multiply-accumulate instruction is using TF32 inputs.
        const INPUT_TF32 = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32 as _;
        /// Specify that the implementation's inner dot product multiply-accumulate instruction is using single-precision inputs.
        const INPUT_32F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_32F as _;
        /// Specify that the implementation's inner dot product multiply-accumulate instruction is using double-precision inputs.
        const INPUT_64F = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_64F as _;
        /// Specify that the implementation's inner dot product multiply-accumulate instruction is using 8-bit integer inputs.
        const INPUT_8I = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8I as _;
        /// Numerical implementation flag identifying 8-bit floating-point inputs in E4M3 format.
        const INPUT_8F_E4M3 = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E4M3 as _;
        /// Numerical implementation flag identifying 8-bit floating-point inputs in E5M2 format.
        const INPUT_8F_E5M2 = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_8F_E5M2 as _;
        /// Mask to filter implementation details about accumulator input used.
        const OP_INPUT_TYPE_MASK = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_OP_INPUT_TYPE_MASK as _;
        /// Specify that the implementation applies Gauss complexity reduction algorithm to reduce arithmetic complexity of the complex matrix multiplication problem
        const GAUSSIAN = sys::CUBLASLT_NUMERICAL_IMPL_FLAGS_GAUSSIAN as _;
    }
}

impl Display for NumericalImplFlags {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        parser::to_writer(self, f)
    }
}

/// Experimental policy used to request batch-invariant matrix multiplication results.
///
/// The alignment choices are mutually exclusive, not combinable bit flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum BatchInvariance {
    /// Batch invariance disabled.
    Disabled = sys::cublasLtMatmulBatchInvarianceFlags_t::CUBLASLT_BATCH_INVARIANCE_DISABLED as _,
    /// Batch invariance assumes 2-byte alignment for tensors.
    Align2B = sys::cublasLtMatmulBatchInvarianceFlags_t::CUBLASLT_BATCH_INVARIANCE_ALIGN_2B as _,
    /// Batch invariance assumes 4-byte alignment for tensors.
    Align4B = sys::cublasLtMatmulBatchInvarianceFlags_t::CUBLASLT_BATCH_INVARIANCE_ALIGN_4B as _,
    /// Batch invariance assumes 8-byte alignment for tensors.
    Align8B = sys::cublasLtMatmulBatchInvarianceFlags_t::CUBLASLT_BATCH_INVARIANCE_ALIGN_8B as _,
    /// Batch invariance assumes 16-byte alignment for tensors.
    Align16B = sys::cublasLtMatmulBatchInvarianceFlags_t::CUBLASLT_BATCH_INVARIANCE_ALIGN_16B as _,
    /// The alignment class is detected from the shape, pointers and minimum alignment
    /// preference attributes of each call. Results are batch invariant only if every
    /// call detects the same class.
    Auto = sys::cublasLtMatmulBatchInvarianceFlags_t::CUBLASLT_BATCH_INVARIANCE_AUTO as _,
    /// Not set through the attribute; batch invariance is controlled by the
    /// `CUBLAS_BATCH_INVARIANCE_FLAGS` environment variable. This is the default value.
    Unset = sys::cublasLtMatmulBatchInvarianceFlags_t::CUBLASLT_BATCH_INVARIANCE_UNSET as _,
}

impl_enum_conversion!(sys::cublasLtMatmulBatchInvarianceFlags_t, BatchInvariance);

impl_enum_display!(BatchInvariance, {
    Self::Disabled => "CUBLASLT_BATCH_INVARIANCE_DISABLED",
    Self::Align2B => "CUBLASLT_BATCH_INVARIANCE_ALIGN_2B",
    Self::Align4B => "CUBLASLT_BATCH_INVARIANCE_ALIGN_4B",
    Self::Align8B => "CUBLASLT_BATCH_INVARIANCE_ALIGN_8B",
    Self::Align16B => "CUBLASLT_BATCH_INVARIANCE_ALIGN_16B",
    Self::Auto => "CUBLASLT_BATCH_INVARIANCE_AUTO",
    Self::Unset => "CUBLASLT_BATCH_INVARIANCE_UNSET",
});
