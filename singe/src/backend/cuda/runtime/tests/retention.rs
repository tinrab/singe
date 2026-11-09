use std::{mem, sync::Arc};

use super::*;
use crate::backend::cuda::runtime::{CublasLtMatmulKeyParts, CudaRuntime, CudaRuntimeCacheLimits};

struct ResetPolicy(Arc<CudaRuntime>);

impl Drop for ResetPolicy {
    fn drop(&mut self) {
        self.0
            .set_provider_cache_limits(CudaRuntimeCacheLimits::default())
            .unwrap();
        self.0.clear_cache().unwrap();
    }
}

#[test]
fn test_positive_negative_and_autotune_metadata_caches_each_enforce_their_limits() {
    let _lock = maybe_lock_device().expect("cache retention requires CUDA device 0");
    let runtime = runtime_for_device(Device::Cuda(0)).unwrap();
    let _reset = ResetPolicy(Arc::clone(&runtime));
    runtime.clear_cache().unwrap();
    let mut limits = CudaRuntimeCacheLimits::default();
    limits.cublaslt.entries = 2;
    limits.cublaslt.workspace_bytes = 2048;
    limits.unsupported_cublaslt.entries = 2;
    limits.autotune.entries = 2;
    limits.autotune.workspace_bytes = 1024;
    runtime.set_provider_cache_limits(limits).unwrap();
    let before = runtime.provider_cache_stats().unwrap();
    // SAFETY: this plain C algorithm record accepts zero bytes.
    // It is used only as cached metadata in this test, never submitted to cuBLASLt.
    let algorithm = unsafe { mem::zeroed::<MatmulAlgorithm>() };
    for rows in 1..17 {
        let key = CublasLtMatmulAlgorithmKey::new(CublasLtMatmulKeyParts {
            data_type: DataType::F32,
            compute_type: ComputeType::F32,
            transpose_a: false,
            transpose_b: false,
            epilogue: Epilogue::Default,
            rows,
            inner: 3,
            cols: 2,
            batch_count: 1,
            a_strided_batch_offset: 0,
            b_strided_batch_offset: 0,
            workspace_limit_bytes: 1024,
            library_version: "retention-test".to_owned(),
        });
        runtime.cublaslt_matmul_algorithms().lock().unwrap().insert(
            key.clone(),
            CublasLtMatmulAlgorithmEntry::new(algorithm, ComputeType::F32, 1024, 1., false, false),
        );
        runtime
            .unsupported_cublaslt_matmul_algorithms()
            .lock()
            .unwrap()
            .insert(key, ());
        let bucket = AutotuneTensorBucket::new(
            DataType::F32,
            vec![i64::try_from(rows).unwrap(), 3],
            vec![3, 1],
            0,
        );
        let key = AutotuneCacheKey::new(
            Operation::MatMul,
            DataType::F32,
            vec![bucket],
            CudaGraphProvider::CublasLt,
            "retention-test".into(),
            Some(1024),
        );
        runtime
            .insert_autotune_cache_entry(key, AutotuneCacheEntry::new(512))
            .unwrap();
        let stats = runtime.provider_cache_stats().unwrap();
        assert!(stats.cublaslt.entries <= 2 && stats.cublaslt.workspace_bytes <= 2048);
        assert!(stats.unsupported_cublaslt.entries <= 2);
        assert!(stats.autotune.entries <= 2 && stats.autotune.workspace_bytes <= 1024);
    }
    let after = runtime.provider_cache_stats().unwrap();
    assert_eq!(after.cublaslt.evictions - before.cublaslt.evictions, 14);
    assert_eq!(
        after.unsupported_cublaslt.evictions - before.unsupported_cublaslt.evictions,
        14
    );
    assert_eq!(after.autotune.evictions - before.autotune.evictions, 14);
    // Shrinking the metadata budget also evicts entries immediately.
    limits.cublaslt.accounted_bytes = 1;
    limits.unsupported_cublaslt.accounted_bytes = 1;
    limits.autotune.accounted_bytes = 1;
    runtime.set_provider_cache_limits(limits).unwrap();
    let after = runtime.provider_cache_stats().unwrap();
    assert_eq!(
        after.cublaslt.entries + after.unsupported_cublaslt.entries + after.autotune.entries,
        0
    );
}
