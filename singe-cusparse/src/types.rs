use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_core::impl_enum_display;

use singe_cuda::view::NoUninitDeviceRepr;

use singe_cusparse_sys as sys;

singe_core::native_enum! {
    /// Indicates whether scalar values are read from host memory or device memory.
    ///
    /// If an operation uses several scalar values, all of them use the same pointer mode.
    /// The pointer mode can be set and retrieved with [`Context::set_scalar_pointer_mode`](crate::context::Context::set_scalar_pointer_mode) and [`Context::scalar_pointer_mode`](crate::context::Context::scalar_pointer_mode), respectively.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum PointerMode: sys::cusparsePointerMode_t {
        /// Scalars are read from host memory.
        Host = CUSPARSE_POINTER_MODE_HOST,
        /// Scalars are read from device memory.
        Device = CUSPARSE_POINTER_MODE_DEVICE,
    }
}

singe_core::native_enum! {
    /// Selects whether an operation processes only indices or both data and indices.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Action: sys::cusparseAction_t {
        /// Process only indices.
        Symbolic = CUSPARSE_ACTION_SYMBOLIC,
        /// Process data and indices.
        Numeric = CUSPARSE_ACTION_NUMERIC,
    }
}

singe_core::native_enum! {
    /// Describes the matrix kind stored in sparse storage.
    /// For symmetric, Hermitian, and triangular matrices, only their lower or upper part is assumed to be stored.
    ///
    /// Matrix type and fill mode minimize storage for symmetric or Hermitian matrices and let `SpMV` operations use symmetry when useful.
    /// To compute `y=A*x` when `A` is symmetric and only lower triangular part is stored, two steps are needed.
    /// First step is to compute `y=(L+D)*x` and second step is to compute `y=L^T*x + y`.
    /// Because the transpose operation `y=L^T*x` is much slower than the non-transpose version `y=L*x`, the symmetric property does not always improve performance.
    /// It is usually more efficient to expand the symmetric matrix to a general matrix and apply `y=A*x` with matrix type [`MatrixType::General`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum MatrixType: sys::cusparseMatrixType_t {
        /// The matrix is general.
        General = CUSPARSE_MATRIX_TYPE_GENERAL,
        /// The matrix is symmetric.
        Symmetric = CUSPARSE_MATRIX_TYPE_SYMMETRIC,
        /// The matrix is Hermitian.
        Hermitian = CUSPARSE_MATRIX_TYPE_HERMITIAN,
        /// The matrix is triangular.
        Triangular = CUSPARSE_MATRIX_TYPE_TRIANGULAR,
    }
}

singe_core::native_enum! {
    /// Selects whether the lower or upper part of a matrix is stored in sparse storage.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum FillMode: sys::cusparseFillMode_t {
        /// The lower triangular part is stored.
        Lower = CUSPARSE_FILL_MODE_LOWER,
        /// The upper triangular part is stored.
        Upper = CUSPARSE_FILL_MODE_UPPER,
    }
}

singe_core::native_enum! {
    /// Selects whether matrix diagonal entries are treated as unity.
    ///
    /// The diagonal elements are always assumed to be present, but if [`DiagonalType::Unit`] is passed to an operation, the operation assumes that all diagonal entries are one and does not read or modify them.
    /// This behavior is independent of the actual values stored in memory.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum DiagonalType: sys::cusparseDiagType_t {
        /// The matrix diagonal has non-unit elements.
        NonUnit = CUSPARSE_DIAG_TYPE_NON_UNIT,
        /// The matrix diagonal has unit elements.
        Unit = CUSPARSE_DIAG_TYPE_UNIT,
    }
}

singe_core::native_enum! {
    /// Selects whether matrix indices are zero-based or one-based.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum IndexBase: sys::cusparseIndexBase_t {
        /// The base index is zero (C compatibility).
        Zero = CUSPARSE_INDEX_BASE_ZERO,
        /// The base index is one (one-based indexing compatibility).
        One = CUSPARSE_INDEX_BASE_ONE,
    }
}

singe_core::native_enum! {
    /// Selects the operation applied to an input, such as a sparse matrix or vector.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Operation: sys::cusparseOperation_t {
        /// Non-transpose operation.
        NonTranspose = CUSPARSE_OPERATION_NON_TRANSPOSE,
        /// Transpose operation.
        Transpose = CUSPARSE_OPERATION_TRANSPOSE,
        /// Conjugate transpose operation.
        ConjugateTranspose = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
    }
}

impl Operation {
    /// Alias for [`Self::ConjugateTranspose`].
    pub const HERMITIAN: Self = Self::ConjugateTranspose;
}

singe_core::native_enum! {
    /// Selects whether dense matrix elements are scanned by rows or columns in `cusparse[S|D|C|Z]nnz`.
    /// This also controls block storage format in BSR matrices.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Direction: sys::cusparseDirection_t {
        /// Matrix elements are scanned by rows.
        Row = CUSPARSE_DIRECTION_ROW,
        /// Matrix elements are scanned by columns.
        Column = CUSPARSE_DIRECTION_COLUMN,
    }
}

singe_core::native_enum! {
    /// Native sparse graph-coloring algorithm selection.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum ColorAlgorithm: sys::cusparseColorAlg_t {
        /// default
        Algorithm0 = CUSPARSE_COLOR_ALG0,
        /// Selects native coloring algorithm variant 1.
        Algorithm1 = CUSPARSE_COLOR_ALG1,
    }
}

singe_core::native_enum! {
    /// Algorithm for converting compressed sparse row storage to compressed sparse column storage.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum CsrToCscAlgorithm: sys::cusparseCsr2CscAlg_t {
        /// Uses the native default CSR-to-CSC conversion algorithm.
        Default = CUSPARSE_CSR2CSC_ALG_DEFAULT,
    }
}

singe_core::native_enum! {
    /// Describes the sparse matrix storage format.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Format: sys::cusparseFormat_t {
        /// The matrix is stored in Compressed Sparse Row (CSR) format.
        Csr = CUSPARSE_FORMAT_CSR,
        /// The matrix is stored in Compressed Sparse Column (CSC) format.
        Csc = CUSPARSE_FORMAT_CSC,
        /// The matrix is stored in Coordinate (COO) format organized in *Structure of Arrays (`SoA`)* layout.
        Coo = CUSPARSE_FORMAT_COO,
        /// The matrix is stored in Blocked-Ellpack (Blocked-ELL) format.
        BlockedEll = CUSPARSE_FORMAT_BLOCKED_ELL,
        /// The matrix is stored in Block Sparse Row (BSR) format.
        Bsr = CUSPARSE_FORMAT_BSR,
        /// Sliced ELL
        SlicedEllpack = CUSPARSE_FORMAT_SLICED_ELLPACK,
    }
}

singe_core::native_enum! {
    /// Describes the memory layout of a dense matrix.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Order: sys::cusparseOrder_t {
        /// The matrix is stored in column-major.
        Column = CUSPARSE_ORDER_COL,
        /// The matrix is stored in row-major.
        Row = CUSPARSE_ORDER_ROW,
    }
}

singe_core::native_enum! {
    /// Describes the integer type used for sparse matrix indices.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum IndexType: sys::cusparseIndexType_t {
        /// 16-bit unsigned integer.
        U16 = CUSPARSE_INDEX_16U,
        /// 32-bit signed integer \[0, 2^31 - 1\].
        I32 = CUSPARSE_INDEX_32I,
        /// 64-bit signed integer \[0, 2^63 - 1\].
        I64 = CUSPARSE_INDEX_64I,
    }
}

singe_core::native_enum! {
    /// Sparse matrix attributes configuring the stored triangle and implicit unit diagonal.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SparseMatrixAttribute: sys::cusparseSpMatAttribute_t {
        /// Indicates if the lower or upper part of a matrix is stored in sparse storage
        FillMode = CUSPARSE_SPMAT_FILL_MODE,
        /// Indicates if the matrix diagonal entries are unity
        DiagonalType = CUSPARSE_SPMAT_DIAG_TYPE,
    }
}

/// Associates a device index representation with its native cuSPARSE datatype.
///
/// # Safety
///
/// The type must have the size, alignment, and integer representation selected by `index_type`, with no padding or restricted integer bit patterns.
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

singe_core::native_enum! {
    /// Algorithm for multiplying a sparse matrix by a dense vector.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SpMvAlgorithm: sys::cusparseSpMVAlg_t {
        /// Default algorithm for any sparse matrix format.
        Default = CUSPARSE_SPMV_ALG_DEFAULT,
        /// Default algorithm for CSR/CSC sparse matrix format.
        /// May produce slightly different results during different runs with the same input parameters.
        Csr1 = CUSPARSE_SPMV_CSR_ALG1,
        /// Provides deterministic (bit-wise) results for each run.
        /// If `opA != CUSPARSE_OPERATION_NON_TRANSPOSE`, it is identical to `CUSPARSE_SPMV_CSR_ALG1`.
        Csr2 = CUSPARSE_SPMV_CSR_ALG2,
        /// Default algorithm for COO sparse matrix format.
        /// May produce slightly different results during different runs with the same input parameters.
        Coo1 = CUSPARSE_SPMV_COO_ALG1,
        /// Provides deterministic (bit-wise) results for each run.
        /// If `opA != CUSPARSE_OPERATION_NON_TRANSPOSE`, it is identical to `CUSPARSE_SPMV_COO_ALG1`.
        Coo2 = CUSPARSE_SPMV_COO_ALG2,
        /// Default algorithm for Sliced Ellpack sparse matrix format.
        /// Provides deterministic (bit-wise) results for each run.
        Sell1 = CUSPARSE_SPMV_SELL_ALG1,
        /// Default algorithm for BSR sparse matrix format.
        /// Provides deterministic (bit-wise) results for each run.
        /// Supports only `opA == CUSPARSE_OPERATION_NON_TRANSPOSE`.
        /// Supports both row-major and column-major block layouts in `A`.
        Bsr1 = CUSPARSE_SPMV_BSR_ALG1,
    }
}

singe_core::native_enum! {
    /// Algorithm for solving a sparse triangular system with one right-hand side.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SpSvAlgorithm: sys::cusparseSpSVAlg_t {
        /// Uses the default sparse triangular vector-solve algorithm.
        Default = CUSPARSE_SPSV_ALG_DEFAULT,
    }
}

singe_core::native_enum! {
    /// Selects whether a sparse triangular vector-solve update replaces all matrix values or only diagonal values.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SpSvUpdate: sys::cusparseSpSVUpdate_t {
        /// Updates the sparse matrix values with values of `newValues` array
        General = CUSPARSE_SPSV_UPDATE_GENERAL,
        /// Updates the diagonal part of the matrix with diagonal values stored in `newValues` array.
        /// That is, `newValues` has the new diagonal values only
        Diagonal = CUSPARSE_SPSV_UPDATE_DIAGONAL,
    }
}

singe_core::native_enum! {
    /// Algorithm for solving a sparse triangular system with a dense matrix of right-hand sides.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SpSmAlgorithm: sys::cusparseSpSMAlg_t {
        /// Uses the default sparse triangular matrix-solve algorithm.
        Default = CUSPARSE_SPSM_ALG_DEFAULT,
    }
}

singe_core::native_enum! {
    /// Selects whether a sparse triangular matrix-solve update replaces all matrix values or only diagonal values.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SpSmUpdate: sys::cusparseSpSMUpdate_t {
        /// Updates the sparse matrix values with values of `newValues` array
        General = CUSPARSE_SPSM_UPDATE_GENERAL,
        /// Updates the diagonal part of the matrix with diagonal values stored in `newValues` array.
        /// That is, `newValues` has the new diagonal values only
        Diagonal = CUSPARSE_SPSM_UPDATE_DIAGONAL,
    }
}

singe_core::native_enum! {
    /// Algorithm for expanding sparse matrix storage into a dense matrix.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SparseToDenseAlgorithm: sys::cusparseSparseToDenseAlg_t {
        /// Uses the default sparse-to-dense conversion algorithm.
        Default = CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
    }
}

singe_core::native_enum! {
    /// Algorithm for selecting and storing nonzero entries of a dense matrix.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum DenseToSparseAlgorithm: sys::cusparseDenseToSparseAlg_t {
        /// Uses the default dense-to-sparse conversion algorithm.
        Default = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
    }
}

singe_core::native_enum! {
    /// Algorithm for multiplying a sparse matrix by a dense matrix.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SpMmAlgorithm: sys::cusparseSpMMAlg_t {
        /// Default algorithm for any sparse matrix format
        Default = CUSPARSE_SPMM_ALG_DEFAULT,
        /// Algorithm 1 for COO sparse matrix format * May provide better performance for small number of nnz * Provides the best performance with column-major layout * It supports batched computation * May produce slightly different results during different runs with the same input parameters
        Coo1 = CUSPARSE_SPMM_COO_ALG1,
        /// Algorithm 2 for COO sparse matrix format * It provides deterministic result * Provides the best performance with column-major layout * In general, slower than Algorithm 1 * It supports batched computation * It requires additional memory * If `opA != CUSPARSE_OPERATION_NON_TRANSPOSE`, it is identical to `CUSPARSE_SPMM_COO_ALG1`
        Coo2 = CUSPARSE_SPMM_COO_ALG2,
        /// Algorithm 3 for COO sparse matrix format * May provide better performance for large number of nnz * May produce slightly different results during different runs with the same input parameters
        Coo3 = CUSPARSE_SPMM_COO_ALG3,
        /// Algorithm 4 for COO sparse matrix format * Provides better performance with row-major layout * It supports batched computation * May produce slightly different results during different runs with the same input parameters
        Coo4 = CUSPARSE_SPMM_COO_ALG4,
        /// Algorithm 1 for CSR/CSC sparse matrix format * Provides the best performance with column-major layout * It supports batched computation * It requires additional memory * May produce slightly different results during different runs with the same input parameters
        Csr1 = CUSPARSE_SPMM_CSR_ALG1,
        /// Algorithm 2 for CSR/CSC sparse matrix format * Provides the best performance with row-major layout * It supports batched computation * It requires additional memory * May produce slightly different results during different runs with the same input parameters
        Csr2 = CUSPARSE_SPMM_CSR_ALG2,
        /// Algorithm 3 for CSR sparse matrix format * It provides deterministic result * It requires additional memory * It supports only CSR matrix and `opA == CUSPARSE_OPERATION_NON_TRANSPOSE` * It does not support `opB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE` * It does not support `CUDA_C_16F and CUDA_C_16BF` data types
        Csr3 = CUSPARSE_SPMM_CSR_ALG3,
        /// Default algorithm for Blocked-ELL format
        BlockedEll1 = CUSPARSE_SPMM_BLOCKED_ELL_ALG1,
        /// Algorithm 1 for BSR sparse matrix format * It provides deterministic result * It requires no additional memory * It supports only `opA == CUSPARSE_OPERATION_NON_TRANSPOSE` * It does not support `CUDA_C_16F` and `CUDA_C_16BF` data types * It does not support column-major blocks in `A`
        Bsr1 = CUSPARSE_SPMM_BSR_ALG1,
    }
}

singe_core::native_enum! {
    /// Algorithm selection for sparse-dense multiplication with custom device operators.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SpMmOpAlgorithm: sys::cusparseSpMMOpAlg_t {
        /// Default algorithm for any sparse matrix format
        Default = CUSPARSE_SPMM_OP_ALG_DEFAULT,
    }
}

singe_core::native_enum! {
    /// Algorithm for multiplying two sparse matrices, including workspace and determinism choices.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SpGemmAlgorithm: sys::cusparseSpGEMMAlg_t {
        /// Default algorithm.
        /// It selects [`SpGemmAlgorithm::Algorithm1`].
        Default = CUSPARSE_SPGEMM_DEFAULT,
        /// Provides deterministic (bit-wise) structure for the output matrix and value computation for each run.
        CsrDeterministic = CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC,
        /// Selects the native non-deterministic CSR sparse matrix multiplication algorithm.
        CsrNondeterministic = CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC,
        /// Computes the product in one pass with deterministic results and the best performance of the three numbered algorithms.
        ///
        /// It calls [`spgemm_compute_async`](crate::spgemm::spgemm_compute_async) twice.
        /// The first call returns an upper bound of the workspace size, which is often several times the memory actually used.
        /// The second call accepts a smaller workspace and fails with [`Status::InsufficientResources`](crate::error::Status::InsufficientResources) if it is too small.
        Algorithm1 = CUSPARSE_SPGEMM_ALG1,
        /// Computes the product with less memory than [`SpGemmAlgorithm::Algorithm1`] and deterministic results.
        ///
        /// It queries the required memory with [`spgemm_estimate_memory_async`](crate::spgemm::spgemm_estimate_memory_async).
        /// It is slower than [`SpGemmAlgorithm::Algorithm1`] and faster than [`SpGemmAlgorithm::Algorithm3`].
        Algorithm2 = CUSPARSE_SPGEMM_ALG2,
        /// Computes the intermediate products in chunks, one chunk at a time, with deterministic results.
        ///
        /// It queries the required memory with [`spgemm_estimate_memory_async`](crate::spgemm::spgemm_estimate_memory_async).
        /// Its `chunk_fraction` argument sets the chunk size as a fraction of the intermediate product count, which bounds the memory used.
        Algorithm3 = CUSPARSE_SPGEMM_ALG3,
    }
}

singe_core::native_enum! {
    /// Algorithm for dense matrix multiplication sampled at the nonzero positions of a sparse matrix.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum SddmmAlgorithm: sys::cusparseSDDMMAlg_t {
        /// Default algorithm.
        /// It supports batched computation.
        Default = CUSPARSE_SDDMM_ALG_DEFAULT,
    }
}

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

impl_enum_display!(SpSmAlgorithm, {
    SpSmAlgorithm::Default => "CUSPARSE_SPSM_ALG_DEFAULT",
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
