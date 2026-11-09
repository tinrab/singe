use singe_cuda::module::KernelFunction;
use singe_cufft::plan::Plan;

use crate::backend::cuda::{
    cudnn_backend::{CudnnBackendGraphCacheEntry, CudnnBackendGraphKey},
    cutensor::{CachedCutensorContractionPlan, CutensorContractionPlanKey},
    runtime::{
        AutotuneCacheEntry, AutotuneCacheKey, CublasLtMatmulAlgorithmEntry,
        CublasLtMatmulAlgorithmKey, FftPlanKey, retention::CacheWeight,
    },
};

// Keys are retained in the lookup map and the FIFO.
// These are conservative charges for keys and entry records, excluding container overhead, native plan metadata and opaque native allocations.
// Saturation conservatively rejects admission if the accounting cannot be represented.
fn charge(key_bytes: usize, value_bytes: usize, workspace_bytes: usize) -> CacheWeight {
    CacheWeight {
        accounted_bytes: key_bytes.saturating_mul(2).saturating_add(value_bytes),
        workspace_bytes,
    }
}

pub fn cudnn(key: &CudnnBackendGraphKey, value: &CudnnBackendGraphCacheEntry) -> CacheWeight {
    let (extra, workspace_bytes) = match value {
        CudnnBackendGraphCacheEntry::Supported { workspace_size, .. } => (0, *workspace_size),
        CudnnBackendGraphCacheEntry::Unsupported { reason } => (reason.capacity(), 0),
    };
    charge(
        key.accounted_bytes(),
        size_of_val(value).saturating_add(extra),
        workspace_bytes,
    )
}

pub fn cutensor(
    key: &CutensorContractionPlanKey,
    value: &CachedCutensorContractionPlan,
) -> CacheWeight {
    charge(
        key.accounted_bytes(),
        size_of_val(value),
        value.workspace_size(),
    )
}

pub fn fft(key: &FftPlanKey, value: &Plan) -> CacheWeight {
    charge(size_of_val(key), size_of_val(value), 0)
}

pub fn cublaslt(
    key: &CublasLtMatmulAlgorithmKey,
    value: &CublasLtMatmulAlgorithmEntry,
) -> CacheWeight {
    charge(
        key.accounted_bytes(),
        size_of_val(value),
        value.workspace_bytes(),
    )
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the generic cache weigher requires references to keys and values"
)]
pub fn unsupported_cublaslt(key: &CublasLtMatmulAlgorithmKey, _: &()) -> CacheWeight {
    charge(key.accounted_bytes(), 0, 0)
}

pub fn autotune(key: &AutotuneCacheKey, value: &AutotuneCacheEntry) -> CacheWeight {
    charge(
        key.accounted_bytes(),
        size_of::<AutotuneCacheEntry>(),
        value.workspace_bytes(),
    )
}

pub fn compiled_kernel(key: &String, value: &KernelFunction) -> CacheWeight {
    charge(
        size_of_val(key).saturating_add(key.capacity()),
        size_of_val(value),
        0,
    )
}
