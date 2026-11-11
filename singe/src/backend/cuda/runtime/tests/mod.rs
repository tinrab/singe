mod allocator;
mod autotune;
mod cublaslt_selection;
mod provider_cache_retention;

use singe_cublas::{
    lt::{matmul::MatmulAlgorithm, types::Epilogue},
    types::ComputeType,
};
use singe_cuda::stream::StreamCaptureMode;

use crate::{
    backend::{
        cuda::{
            self, CublasLtMatmulTuningOptions, CudaGraphProvider,
            runtime::{
                AutotuneCacheEntry, AutotuneCacheKey, CublasLtMatmulAlgorithmEntry,
                CublasLtMatmulAlgorithmFingerprint, CublasLtMatmulAlgorithmKey, runtime_for_device,
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
