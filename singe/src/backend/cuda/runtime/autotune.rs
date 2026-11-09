use singe_cublas::types::ComputeType;

use crate::{
    backend::{
        cuda::{
            compiled::plan::lowering::CudaGraphProvider,
            runtime::CublasLtMatmulAlgorithmFingerprint, tensor_region::CudaTensorSource,
        },
        registry::Operation,
    },
    data_type::DataType,
    graph::NumericalPolicy,
};

/// Tensor data type, dimensions, element strides, and storage offset used in an autotune key.
/// The bucket describes metadata without retaining tensor storage.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AutotuneTensorBucket {
    data_type: DataType,
    dimensions: Vec<i64>,
    strides: Vec<i64>,
    offset: usize,
}

#[cfg(test)]
impl AutotuneTensorBucket {
    #[must_use]
    pub fn new(
        data_type: DataType,
        dimensions: Vec<i64>,
        strides: Vec<i64>,
        offset: usize,
    ) -> Self {
        Self {
            data_type,
            dimensions,
            strides,
            offset,
        }
    }
}

impl AutotuneTensorBucket {
    /// Copies tensor metadata into an autotune bucket without accessing its values.
    pub fn from_tensor(tensor: &(impl CudaTensorSource + ?Sized)) -> Self {
        Self {
            data_type: tensor.data_type(),
            dimensions: tensor.shape().dimensions().to_vec(),
            strides: tensor.layout().strides().to_vec(),
            offset: tensor.storage_offset(),
        }
    }

    /// Copies tensor metadata into buckets in iterator order.
    pub fn from_tensors<'a, S: CudaTensorSource + ?Sized + 'a>(
        tensors: impl IntoIterator<Item = &'a S>,
    ) -> Vec<Self> {
        tensors.into_iter().map(Self::from_tensor).collect()
    }
}

#[cfg(test)]
impl AutotuneTensorBucket {
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    #[must_use]
    pub fn dimensions(&self) -> &[i64] {
        &self.dimensions
    }

    #[must_use]
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    #[must_use]
    pub fn storage_offset(&self) -> usize {
        self.offset
    }
}

/// A tuned algorithm's workspace, compute, fingerprint, and timing metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct AutotuneCacheEntry {
    workspace_bytes: usize,
    compute_type: Option<ComputeType>,
    cublaslt_algorithm_fingerprint: Option<CublasLtMatmulAlgorithmFingerprint>,
    average_elapsed_ms: Option<f32>,
}

impl AutotuneCacheEntry {
    #[cfg(test)]
    #[must_use]
    pub fn new(workspace_bytes: usize) -> Self {
        Self {
            workspace_bytes,
            compute_type: None,
            cublaslt_algorithm_fingerprint: None,
            average_elapsed_ms: None,
        }
    }

    #[must_use]
    pub fn new_cublaslt_matmul_tuning(
        compute_type: ComputeType,
        workspace_bytes: usize,
        cublaslt_algorithm_fingerprint: Option<CublasLtMatmulAlgorithmFingerprint>,
        average_elapsed_ms: f32,
    ) -> Self {
        Self {
            workspace_bytes,
            compute_type: Some(compute_type),
            cublaslt_algorithm_fingerprint,
            average_elapsed_ms: Some(average_elapsed_ms),
        }
    }

    /// Returns the selected algorithm's workspace requirement in bytes.
    #[must_use]
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// Returns the recorded compute type, if available.
    #[must_use]
    pub fn compute_type(&self) -> Option<ComputeType> {
        self.compute_type
    }

    /// Returns the cuBLASLt algorithm fingerprint, if recorded.
    #[must_use]
    pub fn cublaslt_algorithm_fingerprint(&self) -> Option<CublasLtMatmulAlgorithmFingerprint> {
        self.cublaslt_algorithm_fingerprint
    }

    /// Returns the recorded average execution time in milliseconds, if available.
    #[must_use]
    pub fn average_elapsed_ms(&self) -> Option<f32> {
        self.average_elapsed_ms
    }
}

/// Identity for a CUDA autotune result, including numerical policy, operation, tensor metadata, provider, library version, and optional workspace limit.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AutotuneCacheKey {
    numerics: NumericalPolicy,
    operation: Operation,
    data_type: DataType,
    shape_layout_bucket: Vec<AutotuneTensorBucket>,
    provider: CudaGraphProvider,
    library_version: String,
    workspace_limit_bytes: Option<usize>,
}

impl AutotuneCacheKey {
    pub(crate) fn accounted_bytes(&self) -> usize {
        let base = size_of::<Self>()
            .saturating_add(self.library_version.capacity())
            .saturating_add(
                self.shape_layout_bucket
                    .capacity()
                    .saturating_mul(size_of::<AutotuneTensorBucket>()),
            );
        self.shape_layout_bucket
            .iter()
            .flat_map(|bucket| [bucket.dimensions.capacity(), bucket.strides.capacity()])
            .fold(base, |bytes, capacity| {
                bytes.saturating_add(capacity.saturating_mul(size_of::<i64>()))
            })
    }

    /// Creates an autotune key from ordered tensor metadata and the current numerical policy.
    /// The optional workspace limit is measured in bytes.
    /// Tensor values and allocation identities are not part of the key.
    pub fn from_tensors<'a, S: CudaTensorSource + ?Sized + 'a>(
        operation: Operation,
        data_type: DataType,
        tensors: impl IntoIterator<Item = &'a S>,
        provider: CudaGraphProvider,
        library_version: impl Into<String>,
        workspace_limit_bytes: Option<usize>,
    ) -> Self {
        Self {
            operation,
            data_type,
            shape_layout_bucket: AutotuneTensorBucket::from_tensors(tensors),
            numerics: NumericalPolicy::current(),
            provider,
            library_version: library_version.into(),
            workspace_limit_bytes,
        }
    }

    /// Returns the optional workspace byte limit recorded in this key.
    #[must_use]
    pub fn workspace_limit_bytes(&self) -> Option<usize> {
        self.workspace_limit_bytes
    }
}

#[cfg(test)]
impl AutotuneCacheKey {
    #[must_use]
    pub fn new(
        operation: Operation,
        data_type: DataType,
        shape_layout_bucket: Vec<AutotuneTensorBucket>,
        provider: CudaGraphProvider,
        library_version: String,
        workspace_limit_bytes: Option<usize>,
    ) -> Self {
        Self {
            operation,
            data_type,
            shape_layout_bucket,
            numerics: NumericalPolicy::current(),
            provider,
            library_version,
            workspace_limit_bytes,
        }
    }

    #[must_use]
    pub fn operation(&self) -> Operation {
        self.operation
    }

    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    #[must_use]
    pub fn shape_layout_bucket(&self) -> &[AutotuneTensorBucket] {
        &self.shape_layout_bucket
    }

    #[must_use]
    pub fn provider(&self) -> CudaGraphProvider {
        self.provider
    }

    #[must_use]
    pub fn library_version(&self) -> &str {
        &self.library_version
    }
}
