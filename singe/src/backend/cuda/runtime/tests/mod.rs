mod allocator;
mod autotune;
mod cublaslt_selection;
mod cublaslt_stats;
mod provider_cache_retention;
mod retention;

use singe_cublas::{
    lt::{matmul::MatmulAlgorithm, types::Epilogue},
    types::ComputeType,
};
use singe_cuda::stream::StreamCaptureMode;

use crate::{
    backend::{
        cuda::{
            self, CublasLtMatmulAlgorithmCacheStats, CublasLtMatmulTuningOptions,
            CudaGraphProvider,
            allocator::{CudaAllocator, CudaCachingAllocator},
            runtime::{
                AutotuneCacheEntry, AutotuneCacheKey, AutotuneCacheStats, AutotuneTensorBucket,
                CublasLtMatmulAlgorithmEntry, CublasLtMatmulAlgorithmFingerprint,
                CublasLtMatmulAlgorithmKey, runtime_for_device,
            },
        },
        registry::Operation,
    },
    data_type::DataType,
    device::Device,
    error::Error,
    shape::Shape,
    tensor::Tensor,
    testing::{maybe_lock_device, skip_if_stub},
};
