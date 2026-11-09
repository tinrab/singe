use singe_cublas::{lt::types::Epilogue, types::ComputeType};

use crate::{data_type::DataType, graph::NumericalPolicy};

impl CublasLtMatmulAlgorithmKey {
    pub(crate) fn accounted_bytes(&self) -> usize {
        size_of::<Self>().saturating_add(self.library_version.capacity())
    }

    /// Returns the workspace byte limit used when selecting this cached algorithm.
    #[must_use]
    pub fn workspace_limit_bytes(&self) -> usize {
        self.workspace_limit_bytes
    }

    /// Returns whether both keys describe the same matrix multiplication except for their workspace limits.
    /// The comparison includes the numerical policy, operand and compute types, transposes, epilogue, dimensions, batch layout, and library version.
    #[must_use]
    pub fn matches_matmul_algorithm_family(&self, other: &Self) -> bool {
        self.data_type == other.data_type
            && self.numerics == other.numerics
            && self.compute_type == other.compute_type
            && self.transpose_a == other.transpose_a
            && self.transpose_b == other.transpose_b
            && self.epilogue == other.epilogue
            && self.rows == other.rows
            && self.inner == other.inner
            && self.cols == other.cols
            && self.batch_count == other.batch_count
            && self.a_strided_batch_offset == other.a_strided_batch_offset
            && self.b_strided_batch_offset == other.b_strided_batch_offset
            && self.library_version == other.library_version
    }

    /// Returns the matrix element data type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the cuBLAS compute type used for accumulation.
    #[must_use]
    pub fn compute_type(&self) -> ComputeType {
        self.compute_type
    }

    /// Returns whether matrix A is transposed for multiplication.
    #[must_use]
    pub fn transpose_a(&self) -> bool {
        self.transpose_a
    }

    /// Returns whether matrix B is transposed for multiplication.
    #[must_use]
    pub fn transpose_b(&self) -> bool {
        self.transpose_b
    }

    /// Returns the operation applied after matrix multiplication.
    #[must_use]
    pub fn epilogue(&self) -> Epilogue {
        self.epilogue
    }

    /// Returns the output row count.
    #[must_use]
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Returns the contraction dimension shared by matrices A and B.
    #[must_use]
    pub fn inner(&self) -> u64 {
        self.inner
    }

    /// Returns the output column count.
    #[must_use]
    pub fn cols(&self) -> u64 {
        self.cols
    }

    /// Returns the number of matrix multiplications in the batch.
    #[must_use]
    pub fn batch_count(&self) -> usize {
        self.batch_count
    }

    /// Returns the element stride between consecutive matrices A.
    #[must_use]
    pub fn lhs_strided_batch_offset(&self) -> i64 {
        self.a_strided_batch_offset
    }

    /// Returns the element stride between consecutive matrices B.
    #[must_use]
    pub fn rhs_strided_batch_offset(&self) -> i64 {
        self.b_strided_batch_offset
    }

    /// Returns the library version string included in this cache identity.
    #[must_use]
    pub fn library_version(&self) -> &str {
        &self.library_version
    }
}

/// Cache identity for a cuBLASLt matrix multiplication algorithm selection.
/// The identity includes the active numerical policy and the workspace limit used for selection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CublasLtMatmulAlgorithmKey {
    pub(crate) numerics: NumericalPolicy,
    pub(crate) data_type: DataType,
    pub(crate) compute_type: ComputeType,
    pub(crate) transpose_a: bool,
    pub(crate) transpose_b: bool,
    pub(crate) epilogue: Epilogue,
    pub(crate) rows: u64,
    pub(crate) inner: u64,
    pub(crate) cols: u64,
    pub(crate) batch_count: usize,
    pub(crate) a_strided_batch_offset: i64,
    pub(crate) b_strided_batch_offset: i64,
    pub(crate) workspace_limit_bytes: usize,
    pub(crate) library_version: String,
}

// What identifies a matrix multiplication, before the numerical policy it runs under.
pub struct CublasLtMatmulKeyParts {
    pub data_type: DataType,
    pub compute_type: ComputeType,
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub epilogue: Epilogue,
    pub rows: u64,
    pub inner: u64,
    pub cols: u64,
    pub batch_count: usize,
    pub a_strided_batch_offset: i64,
    pub b_strided_batch_offset: i64,
    pub workspace_limit_bytes: usize,
    pub library_version: String,
}

impl CublasLtMatmulAlgorithmKey {
    /// Creates a matrix multiplication key with the current numerical policy.
    /// Batch offsets are measured in elements, and `workspace_limit_bytes` is measured in bytes.
    /// This constructor stores metadata without validating matrix compatibility or native algorithm support.
    pub fn new(matmul: CublasLtMatmulKeyParts) -> Self {
        Self {
            data_type: matmul.data_type,
            numerics: NumericalPolicy::current(),
            compute_type: matmul.compute_type,
            transpose_a: matmul.transpose_a,
            transpose_b: matmul.transpose_b,
            epilogue: matmul.epilogue,
            rows: matmul.rows,
            inner: matmul.inner,
            cols: matmul.cols,
            batch_count: matmul.batch_count,
            a_strided_batch_offset: matmul.a_strided_batch_offset,
            b_strided_batch_offset: matmul.b_strided_batch_offset,
            workspace_limit_bytes: matmul.workspace_limit_bytes,
            library_version: matmul.library_version,
        }
    }
}
