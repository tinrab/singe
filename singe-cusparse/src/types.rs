use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_core::{impl_enum_conversion, impl_enum_display};

use singe_cuda::view::NoUninitDeviceRepr;

use singe_cusparse_sys as sys;

/// Indicates whether scalar values are read from host memory or device memory.
///
/// If an operation uses several scalar values, all of them use the same pointer mode.
/// The pointer mode can be set and retrieved with [`Context::set_scalar_pointer_mode`](crate::context::Context::set_scalar_pointer_mode) and [`Context::scalar_pointer_mode`](crate::context::Context::scalar_pointer_mode), respectively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum PointerMode {
    /// Scalars are read from host memory.
    Host = sys::cusparsePointerMode_t::CUSPARSE_POINTER_MODE_HOST as _,
    /// Scalars are read from device memory.
    Device = sys::cusparsePointerMode_t::CUSPARSE_POINTER_MODE_DEVICE as _,
}

impl_enum_conversion!(sys::cusparsePointerMode_t, PointerMode);

/// Selects whether an operation processes only indices or both data and indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Action {
    /// Process only indices.
    Symbolic = sys::cusparseAction_t::CUSPARSE_ACTION_SYMBOLIC as _,
    /// Process data and indices.
    Numeric = sys::cusparseAction_t::CUSPARSE_ACTION_NUMERIC as _,
}

impl_enum_conversion!(sys::cusparseAction_t, Action);

/// Describes the matrix kind stored in sparse storage.
/// For symmetric, Hermitian, and triangular matrices, only their lower or upper part is assumed to be stored.
///
/// Matrix type and fill mode minimize storage for symmetric or Hermitian matrices and let `SpMV` operations use symmetry when useful.
/// To compute `y=A*x` when `A` is symmetric and only lower triangular part is stored, two steps are needed.
/// First step is to compute `y=(L+D)*x` and second step is to compute `y=L^T*x + y`.
/// Because the transpose operation `y=L^T*x` is much slower than the non-transpose version `y=L*x`, the symmetric property does not always improve performance.
/// It is usually more efficient to expand the symmetric matrix to a general matrix and
/// apply `y=A*x` with matrix type [`MatrixType::General`].
///
/// In general, `SpMV`, preconditioners (incomplete Cholesky or incomplete LU) and triangular solver are combined together in iterative solvers, for example PCG and GMRES.
/// If applications use general matrices instead of symmetric matrices, preconditioners
/// only need to support general matrices.
/// Therefore the newer `\[bsr|csr\]sv2` (triangular solver), `\[bsr|csr\]ilu02` (incomplete LU), and `\[bsr|csr\]ic02` (incomplete Cholesky) operations only support matrix type [`MatrixType::General`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum MatrixType {
    /// The matrix is general.
    General = sys::cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_GENERAL as _,
    /// The matrix is symmetric.
    Symmetric = sys::cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_SYMMETRIC as _,
    /// The matrix is Hermitian.
    Hermitian = sys::cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_HERMITIAN as _,
    /// The matrix is triangular.
    Triangular = sys::cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_TRIANGULAR as _,
}

impl_enum_conversion!(sys::cusparseMatrixType_t, MatrixType);

/// Selects whether the lower or upper part of a matrix is stored in sparse storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum FillMode {
    /// The lower triangular part is stored.
    Lower = sys::cusparseFillMode_t::CUSPARSE_FILL_MODE_LOWER as _,
    /// The upper triangular part is stored.
    Upper = sys::cusparseFillMode_t::CUSPARSE_FILL_MODE_UPPER as _,
}

impl_enum_conversion!(sys::cusparseFillMode_t, FillMode);

/// Selects whether matrix diagonal entries are treated as unity.
///
/// The diagonal elements are always assumed to be present, but if
/// [`DiagonalType::Unit`] is passed to an operation, the operation assumes that
/// all diagonal entries are one and does not read or modify them.
/// This behavior is independent of the actual values stored in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DiagonalType {
    /// The matrix diagonal has non-unit elements.
    NonUnit = sys::cusparseDiagType_t::CUSPARSE_DIAG_TYPE_NON_UNIT as _,
    /// The matrix diagonal has unit elements.
    Unit = sys::cusparseDiagType_t::CUSPARSE_DIAG_TYPE_UNIT as _,
}

impl_enum_conversion!(sys::cusparseDiagType_t, DiagonalType);

/// Selects whether matrix indices are zero-based or one-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum IndexBase {
    /// The base index is zero (C compatibility).
    Zero = sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO as _,
    /// The base index is one (one-based indexing compatibility).
    One = sys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ONE as _,
}

impl_enum_conversion!(sys::cusparseIndexBase_t, IndexBase);

/// Selects the operation applied to an input, such as a sparse matrix or vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Operation {
    /// Non-transpose operation.
    NonTranspose = sys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE as _,
    /// Transpose operation.
    Transpose = sys::cusparseOperation_t::CUSPARSE_OPERATION_TRANSPOSE as _,
    /// Conjugate transpose operation.
    ConjugateTranspose = sys::cusparseOperation_t::CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE as _,
}

impl Operation {
    /// Alias for [`Self::ConjugateTranspose`].
    pub const HERMITIAN: Self = Self::ConjugateTranspose;
}

impl_enum_conversion!(sys::cusparseOperation_t, Operation);

/// Selects whether dense matrix elements are scanned by rows or columns in `cusparse[S|D|C|Z]nnz`.
/// This also controls block storage format in BSR matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Direction {
    /// Matrix elements are scanned by rows.
    Row = sys::cusparseDirection_t::CUSPARSE_DIRECTION_ROW as _,
    /// Matrix elements are scanned by columns.
    Column = sys::cusparseDirection_t::CUSPARSE_DIRECTION_COLUMN as _,
}

impl_enum_conversion!(sys::cusparseDirection_t, Direction);

/// Native sparse graph-coloring algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum ColorAlgorithm {
    /// default
    Algorithm0 = sys::cusparseColorAlg_t::CUSPARSE_COLOR_ALG0 as _,
    /// Selects native coloring algorithm variant 1.
    Algorithm1 = sys::cusparseColorAlg_t::CUSPARSE_COLOR_ALG1 as _,
}

impl_enum_conversion!(sys::cusparseColorAlg_t, ColorAlgorithm);

/// Algorithm for converting compressed sparse row storage to compressed sparse column storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum CsrToCscAlgorithm {
    /// Uses the native default CSR-to-CSC conversion algorithm.
    Default = sys::cusparseCsr2CscAlg_t::CUSPARSE_CSR2CSC_ALG_DEFAULT as _,
}

impl_enum_conversion!(sys::cusparseCsr2CscAlg_t, CsrToCscAlgorithm);

/// Describes the sparse matrix storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Format {
    /// The matrix is stored in Compressed Sparse Row (CSR) format.
    Csr = sys::cusparseFormat_t::CUSPARSE_FORMAT_CSR as _,
    /// The matrix is stored in Compressed Sparse Column (CSC) format.
    Csc = sys::cusparseFormat_t::CUSPARSE_FORMAT_CSC as _,
    /// The matrix is stored in Coordinate (COO) format organized in *Structure of Arrays (`SoA`)* layout.
    Coo = sys::cusparseFormat_t::CUSPARSE_FORMAT_COO as _,
    /// The matrix is stored in Blocked-Ellpack (Blocked-ELL) format.
    BlockedEll = sys::cusparseFormat_t::CUSPARSE_FORMAT_BLOCKED_ELL as _,
    /// The matrix is stored in Block Sparse Row (BSR) format.
    Bsr = sys::cusparseFormat_t::CUSPARSE_FORMAT_BSR as _,
    /// Sliced ELL
    SlicedEllpack = sys::cusparseFormat_t::CUSPARSE_FORMAT_SLICED_ELLPACK as _,
}

impl_enum_conversion!(sys::cusparseFormat_t, Format);

/// Describes the memory layout of a dense matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Order {
    /// The matrix is stored in column-major.
    Column = sys::cusparseOrder_t::CUSPARSE_ORDER_COL as _,
    /// The matrix is stored in row-major.
    Row = sys::cusparseOrder_t::CUSPARSE_ORDER_ROW as _,
}

impl_enum_conversion!(sys::cusparseOrder_t, Order);

/// Describes the integer type used for sparse matrix indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum IndexType {
    /// 16-bit unsigned integer.
    U16 = sys::cusparseIndexType_t::CUSPARSE_INDEX_16U as _,
    /// 32-bit signed integer \[0, 2^31 - 1\].
    I32 = sys::cusparseIndexType_t::CUSPARSE_INDEX_32I as _,
    /// 64-bit signed integer \[0, 2^63 - 1\].
    I64 = sys::cusparseIndexType_t::CUSPARSE_INDEX_64I as _,
}

impl_enum_conversion!(sys::cusparseIndexType_t, IndexType);

/// Sparse matrix attributes configuring the stored triangle and implicit unit diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SparseMatrixAttribute {
    /// Indicates if the lower or upper part of a matrix is stored in sparse storage
    FillMode = sys::cusparseSpMatAttribute_t::CUSPARSE_SPMAT_FILL_MODE as _,
    /// Indicates if the matrix diagonal entries are unity
    DiagonalType = sys::cusparseSpMatAttribute_t::CUSPARSE_SPMAT_DIAG_TYPE as _,
}

impl_enum_conversion!(sys::cusparseSpMatAttribute_t, SparseMatrixAttribute);

/// Associates a device index representation with its native cuSPARSE datatype.
///
/// # Safety
///
/// The type must have the size, alignment, and integer representation selected
/// by `index_type`, with no padding or restricted integer bit patterns.
pub unsafe trait IndexTypeLike: NoUninitDeviceRepr {
    /// Returns the cuSPARSE index representation associated with this Rust integer type.
    fn index_type() -> IndexType;
}

// SAFETY: u16 has the native unsigned 16-bit index representation.
unsafe impl IndexTypeLike for u16 {
    fn index_type() -> IndexType {
        IndexType::U16
    }
}

// SAFETY: i32 has the native signed 32-bit index representation.
unsafe impl IndexTypeLike for i32 {
    fn index_type() -> IndexType {
        IndexType::I32
    }
}

// SAFETY: i64 has the native signed 64-bit index representation.
unsafe impl IndexTypeLike for i64 {
    fn index_type() -> IndexType {
        IndexType::I64
    }
}

/// Algorithm for multiplying a sparse matrix by a dense vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SpMvAlgorithm {
    /// Default algorithm for any sparse matrix format.
    Default = sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT as _,
    /// Default algorithm for CSR/CSC sparse matrix format. May produce slightly different results during different runs with the same input parameters.
    Csr1 = sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG1 as _,
    /// Provides deterministic (bit-wise) results for each run. If `opA != CUSPARSE_OPERATION_NON_TRANSPOSE`, it is identical to `CUSPARSE_SPMV_CSR_ALG1`.
    Csr2 = sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_CSR_ALG2 as _,
    /// Default algorithm for COO sparse matrix format. May produce slightly different results during different runs with the same input parameters.
    Coo1 = sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_COO_ALG1 as _,
    /// Provides deterministic (bit-wise) results for each run. If `opA != CUSPARSE_OPERATION_NON_TRANSPOSE`, it is identical to `CUSPARSE_SPMV_COO_ALG1`.
    Coo2 = sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_COO_ALG2 as _,
    /// Default algorithm for Sliced Ellpack sparse matrix format. Provides deterministic (bit-wise) results for each run.
    Sell1 = sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_SELL_ALG1 as _,
    /// Default algorithm for BSR sparse matrix format. Provides deterministic (bit-wise) results for each run. Supports only `opA == CUSPARSE_OPERATION_NON_TRANSPOSE`. Supports both row-major and column-major block layouts in `A`.
    Bsr1 = sys::cusparseSpMVAlg_t::CUSPARSE_SPMV_BSR_ALG1 as _,
}

impl_enum_conversion!(sys::cusparseSpMVAlg_t, SpMvAlgorithm);

/// Algorithm for solving a sparse triangular system with one right-hand side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SpSvAlgorithm {
    /// Uses the default sparse triangular vector-solve algorithm.
    Default = sys::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT as _,
}

impl_enum_conversion!(sys::cusparseSpSVAlg_t, SpSvAlgorithm);

/// Selects whether a sparse triangular vector-solve update replaces all matrix values or only diagonal values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SpSvUpdate {
    /// Updates the sparse matrix values with values of `newValues` array
    General = sys::cusparseSpSVUpdate_t::CUSPARSE_SPSV_UPDATE_GENERAL as _,
    /// Updates the diagonal part of the matrix with diagonal values stored in `newValues` array. That is, `newValues` has the new diagonal values only
    Diagonal = sys::cusparseSpSVUpdate_t::CUSPARSE_SPSV_UPDATE_DIAGONAL as _,
}

impl_enum_conversion!(sys::cusparseSpSVUpdate_t, SpSvUpdate);

/// Algorithm for solving a sparse triangular system with a dense matrix of right-hand sides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SpSMAlgorithm {
    /// Uses the default sparse triangular matrix-solve algorithm.
    Default = sys::cusparseSpSMAlg_t::CUSPARSE_SPSM_ALG_DEFAULT as _,
}

impl_enum_conversion!(sys::cusparseSpSMAlg_t, SpSMAlgorithm);

/// Selects whether a sparse triangular matrix-solve update replaces all matrix values or only diagonal values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SpSmUpdate {
    /// Updates the sparse matrix values with values of `newValues` array
    General = sys::cusparseSpSMUpdate_t::CUSPARSE_SPSM_UPDATE_GENERAL as _,
    /// Updates the diagonal part of the matrix with diagonal values stored in `newValues` array. That is, `newValues` has the new diagonal values only
    Diagonal = sys::cusparseSpSMUpdate_t::CUSPARSE_SPSM_UPDATE_DIAGONAL as _,
}

impl_enum_conversion!(sys::cusparseSpSMUpdate_t, SpSmUpdate);

/// Algorithm for expanding sparse matrix storage into a dense matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SparseToDenseAlgorithm {
    /// Uses the default sparse-to-dense conversion algorithm.
    Default = sys::cusparseSparseToDenseAlg_t::CUSPARSE_SPARSETODENSE_ALG_DEFAULT as _,
}

impl_enum_conversion!(sys::cusparseSparseToDenseAlg_t, SparseToDenseAlgorithm);

/// Algorithm for selecting and storing nonzero entries of a dense matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum DenseToSparseAlgorithm {
    /// Uses the default dense-to-sparse conversion algorithm.
    Default = sys::cusparseDenseToSparseAlg_t::CUSPARSE_DENSETOSPARSE_ALG_DEFAULT as _,
}

impl_enum_conversion!(sys::cusparseDenseToSparseAlg_t, DenseToSparseAlgorithm);

/// Algorithm for multiplying a sparse matrix by a dense matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SpMmAlgorithm {
    /// Default algorithm for any sparse matrix format
    Default = sys::cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT as _,
    /// Algorithm 1 for COO sparse matrix format * May provide better performance for small number of nnz * Provides the best performance with column-major layout * It supports batched computation * May produce slightly different results during different runs with the same input parameters
    Coo1 = sys::cusparseSpMMAlg_t::CUSPARSE_SPMM_COO_ALG1 as _,
    /// Algorithm 2 for COO sparse matrix format * It provides deterministic result * Provides the best performance with column-major layout * In general, slower than Algorithm 1 * It supports batched computation * It requires additional memory * If `opA != CUSPARSE_OPERATION_NON_TRANSPOSE`, it is identical to `CUSPARSE_SPMM_COO_ALG1`
    Coo2 = sys::cusparseSpMMAlg_t::CUSPARSE_SPMM_COO_ALG2 as _,
    /// Algorithm 3 for COO sparse matrix format * May provide better performance for large number of nnz * May produce slightly different results during different runs with the same input parameters
    Coo3 = sys::cusparseSpMMAlg_t::CUSPARSE_SPMM_COO_ALG3 as _,
    /// Algorithm 4 for COO sparse matrix format * Provides better performance with row-major layout * It supports batched computation * May produce slightly different results during different runs with the same input parameters
    Coo4 = sys::cusparseSpMMAlg_t::CUSPARSE_SPMM_COO_ALG4 as _,
    /// Algorithm 1 for CSR/CSC sparse matrix format * Provides the best performance with column-major layout * It supports batched computation * It requires additional memory * May produce slightly different results during different runs with the same input parameters
    Csr1 = sys::cusparseSpMMAlg_t::CUSPARSE_SPMM_CSR_ALG1 as _,
    /// Algorithm 2 for CSR/CSC sparse matrix format * Provides the best performance with row-major layout * It supports batched computation * It requires additional memory * May produce slightly different results during different runs with the same input parameters
    Csr2 = sys::cusparseSpMMAlg_t::CUSPARSE_SPMM_CSR_ALG2 as _,
    /// Algorithm 3 for CSR sparse matrix format * It provides deterministic result * It requires additional memory * It supports only CSR matrix and `opA == CUSPARSE_OPERATION_NON_TRANSPOSE` * It does not support `opB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE` * It does not support `CUDA_C_16F and CUDA_C_16BF` data types
    Csr3 = sys::cusparseSpMMAlg_t::CUSPARSE_SPMM_CSR_ALG3 as _,
    /// Default algorithm for Blocked-ELL format
    BlockedEll1 = sys::cusparseSpMMAlg_t::CUSPARSE_SPMM_BLOCKED_ELL_ALG1 as _,
    /// Algorithm 1 for BSR sparse matrix format * It provides deterministic result * It requires no additional memory * It supports only `opA == CUSPARSE_OPERATION_NON_TRANSPOSE` * It does not support `CUDA_C_16F` and `CUDA_C_16BF` data types * It does not support column-major blocks in `A`
    Bsr1 = sys::cusparseSpMMAlg_t::CUSPARSE_SPMM_BSR_ALG1 as _,
}

impl_enum_conversion!(sys::cusparseSpMMAlg_t, SpMmAlgorithm);

/// Algorithm selection for sparse-dense multiplication with custom device operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SpMmOpAlgorithm {
    /// Default algorithm for any sparse matrix format
    Default = sys::cusparseSpMMOpAlg_t::CUSPARSE_SPMM_OP_ALG_DEFAULT as _,
}

impl_enum_conversion!(sys::cusparseSpMMOpAlg_t, SpMmOpAlgorithm);

/// Algorithm for multiplying two sparse matrices, including workspace and determinism choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SpGemmAlgorithm {
    /// Default algorithm. Currently, it is `CUSPARSE_SPGEMM_ALG1`.
    Default = sys::cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_DEFAULT as _,
    /// Provides deterministic (bit-wise) structure for the output matrix and value computation for each run.
    CsrDeterministic = sys::cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC as _,
    /// Selects the native non-deterministic CSR sparse matrix multiplication algorithm.
    CsrNondeterministic = sys::cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC as _,
    /// Algorithm 1 * Invokes `cusparseSpGEMM_compute` twice. The first invocation provides an upper bound of the memory required for the computation. * The required memory is generally several times larger of the actual memory used. * The user can provide an arbitrary buffer size bufferSize2 in the second invocation. If it is not sufficient, the routine will returns `CUSPARSE_STATUS_INSUFFICIENT_RESOURCES` status. * Provides better performance than other algorithms. * Provides deterministic (bit-wise) results for each run.
    Algorithm1 = sys::cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_ALG1 as _,
    /// Algorithm 2 * Invokes `cusparseSpGEMM_estimateMemory` to get the amount of the memory required for the computation. * Requires less memory for the computation than Algorithm 1. * Performance is lower than Algorithm 1, higher than Algorithm 3. * Provides deterministic (bit-wise) results for each run.
    Algorithm2 = sys::cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_ALG2 as _,
    /// Algorithm 3 * Computes the intermediate products in chunks, one chunk at a time. * Invokes `cusparseSpGEMM_estimateMemory` to get the amount of the memory required for the computation. * The user can control the amount of required memory by changing the chunk size via `chunk_fraction`. * The chunk size is a fraction of total intermediate products: `chunk_fraction * (*num_prods)`. * Provides deterministic (bit-wise) results for each run.
    Algorithm3 = sys::cusparseSpGEMMAlg_t::CUSPARSE_SPGEMM_ALG3 as _,
}

impl_enum_conversion!(sys::cusparseSpGEMMAlg_t, SpGemmAlgorithm);

/// Algorithm for dense matrix multiplication sampled at the nonzero positions of a sparse matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SddmmAlgorithm {
    /// Default algorithm. It supports batched computation.
    Default = sys::cusparseSDDMMAlg_t::CUSPARSE_SDDMM_ALG_DEFAULT as _,
}

impl_enum_conversion!(sys::cusparseSDDMMAlg_t, SddmmAlgorithm);

impl_enum_display!(PointerMode, {
    PointerMode::Host => "CUSPARSE_POINTER_MODE_HOST",
    PointerMode::Device => "CUSPARSE_POINTER_MODE_DEVICE",
});

impl_enum_display!(Action, {
    Action::Symbolic => "CUSPARSE_ACTION_SYMBOLIC",
    Action::Numeric => "CUSPARSE_ACTION_NUMERIC",
});

impl_enum_display!(MatrixType, {
    MatrixType::General => "CUSPARSE_MATRIX_TYPE_GENERAL",
    MatrixType::Symmetric => "CUSPARSE_MATRIX_TYPE_SYMMETRIC",
    MatrixType::Hermitian => "CUSPARSE_MATRIX_TYPE_HERMITIAN",
    MatrixType::Triangular => "CUSPARSE_MATRIX_TYPE_TRIANGULAR",
});

impl_enum_display!(FillMode, {
    FillMode::Lower => "CUSPARSE_FILL_MODE_LOWER",
    FillMode::Upper => "CUSPARSE_FILL_MODE_UPPER",
});

impl_enum_display!(DiagonalType, {
    DiagonalType::NonUnit => "CUSPARSE_DIAG_TYPE_NON_UNIT",
    DiagonalType::Unit => "CUSPARSE_DIAG_TYPE_UNIT",
});

impl_enum_display!(IndexBase, {
    IndexBase::Zero => "CUSPARSE_INDEX_BASE_ZERO",
    IndexBase::One => "CUSPARSE_INDEX_BASE_ONE",
});

impl_enum_display!(Operation, {
    Operation::NonTranspose => "CUSPARSE_OPERATION_NON_TRANSPOSE",
    Operation::Transpose => "CUSPARSE_OPERATION_TRANSPOSE",
    Operation::ConjugateTranspose => "CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE",
});

impl_enum_display!(Direction, {
    Direction::Row => "CUSPARSE_DIRECTION_ROW",
    Direction::Column => "CUSPARSE_DIRECTION_COLUMN",
});

impl_enum_display!(ColorAlgorithm, {
    ColorAlgorithm::Algorithm0 => "CUSPARSE_COLOR_ALG0",
    ColorAlgorithm::Algorithm1 => "CUSPARSE_COLOR_ALG1",
});

impl_enum_display!(CsrToCscAlgorithm, {
    CsrToCscAlgorithm::Default => "CUSPARSE_CSR2CSC_ALG_DEFAULT",
});

impl_enum_display!(Format, {
    Format::Csr => "CUSPARSE_FORMAT_CSR",
    Format::Csc => "CUSPARSE_FORMAT_CSC",
    Format::Coo => "CUSPARSE_FORMAT_COO",
    Format::BlockedEll => "CUSPARSE_FORMAT_BLOCKED_ELL",
    Format::Bsr => "CUSPARSE_FORMAT_BSR",
    Format::SlicedEllpack => "CUSPARSE_FORMAT_SLICED_ELLPACK",
});

impl_enum_display!(Order, {
    Order::Column => "CUSPARSE_ORDER_COL",
    Order::Row => "CUSPARSE_ORDER_ROW",
});

impl_enum_display!(IndexType, {
    IndexType::U16 => "CUSPARSE_INDEX_16U",
    IndexType::I32 => "CUSPARSE_INDEX_32I",
    IndexType::I64 => "CUSPARSE_INDEX_64I",
});

impl_enum_display!(SparseMatrixAttribute, {
    SparseMatrixAttribute::FillMode => "CUSPARSE_SPMAT_FILL_MODE",
    SparseMatrixAttribute::DiagonalType => "CUSPARSE_SPMAT_DIAG_TYPE",
});

impl_enum_display!(SpMvAlgorithm, {
    SpMvAlgorithm::Default => "CUSPARSE_SPMV_ALG_DEFAULT",
    SpMvAlgorithm::Csr1 => "CUSPARSE_SPMV_CSR_ALG1",
    SpMvAlgorithm::Csr2 => "CUSPARSE_SPMV_CSR_ALG2",
    SpMvAlgorithm::Coo1 => "CUSPARSE_SPMV_COO_ALG1",
    SpMvAlgorithm::Coo2 => "CUSPARSE_SPMV_COO_ALG2",
    SpMvAlgorithm::Sell1 => "CUSPARSE_SPMV_SELL_ALG1",
    SpMvAlgorithm::Bsr1 => "CUSPARSE_SPMV_BSR_ALG1",
});

impl_enum_display!(SpSvAlgorithm, {
    SpSvAlgorithm::Default => "CUSPARSE_SPSV_ALG_DEFAULT",
});

impl_enum_display!(SpSvUpdate, {
    SpSvUpdate::General => "CUSPARSE_SPSV_UPDATE_GENERAL",
    SpSvUpdate::Diagonal => "CUSPARSE_SPSV_UPDATE_DIAGONAL",
});

impl_enum_display!(SpSMAlgorithm, {
    SpSMAlgorithm::Default => "CUSPARSE_SPSM_ALG_DEFAULT",
});

impl_enum_display!(SpSmUpdate, {
    SpSmUpdate::General => "CUSPARSE_SPSM_UPDATE_GENERAL",
    SpSmUpdate::Diagonal => "CUSPARSE_SPSM_UPDATE_DIAGONAL",
});

impl_enum_display!(SparseToDenseAlgorithm, {
    SparseToDenseAlgorithm::Default => "CUSPARSE_SPARSETODENSE_ALG_DEFAULT",
});

impl_enum_display!(DenseToSparseAlgorithm, {
    DenseToSparseAlgorithm::Default => "CUSPARSE_DENSETOSPARSE_ALG_DEFAULT",
});

impl_enum_display!(SpMmAlgorithm, {
    SpMmAlgorithm::Default => "CUSPARSE_SPMM_ALG_DEFAULT",
    SpMmAlgorithm::Coo1 => "CUSPARSE_SPMM_COO_ALG1",
    SpMmAlgorithm::Coo2 => "CUSPARSE_SPMM_COO_ALG2",
    SpMmAlgorithm::Coo3 => "CUSPARSE_SPMM_COO_ALG3",
    SpMmAlgorithm::Coo4 => "CUSPARSE_SPMM_COO_ALG4",
    SpMmAlgorithm::Csr1 => "CUSPARSE_SPMM_CSR_ALG1",
    SpMmAlgorithm::Csr2 => "CUSPARSE_SPMM_CSR_ALG2",
    SpMmAlgorithm::Csr3 => "CUSPARSE_SPMM_CSR_ALG3",
    SpMmAlgorithm::BlockedEll1 => "CUSPARSE_SPMM_BLOCKED_ELL_ALG1",
    SpMmAlgorithm::Bsr1 => "CUSPARSE_SPMM_BSR_ALG1",
});

impl_enum_display!(SpMmOpAlgorithm, {
    SpMmOpAlgorithm::Default => "CUSPARSE_SPMM_OP_ALG_DEFAULT",
});

impl_enum_display!(SpGemmAlgorithm, {
    SpGemmAlgorithm::Default => "CUSPARSE_SPGEMM_DEFAULT",
    SpGemmAlgorithm::CsrDeterministic => "CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC",
    SpGemmAlgorithm::CsrNondeterministic => "CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC",
    SpGemmAlgorithm::Algorithm1 => "CUSPARSE_SPGEMM_ALG1",
    SpGemmAlgorithm::Algorithm2 => "CUSPARSE_SPGEMM_ALG2",
    SpGemmAlgorithm::Algorithm3 => "CUSPARSE_SPGEMM_ALG3",
});

impl_enum_display!(SddmmAlgorithm, {
    SddmmAlgorithm::Default => "CUSPARSE_SDDMM_ALG_DEFAULT",
});
