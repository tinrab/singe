use std::mem;

use super::*;

use crate::backend::cuda::runtime::CublasLtMatmulKeyParts;

#[test]
fn test_cuda_backend_tuned_cublaslt_selection_respects_recorded_compute_type() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let device = Device::Cuda(0);
    let Some(runtime) = skip_if_stub(runtime_for_device(device)) else {
        return;
    };
    cuda::clear_cache(device).unwrap();

    // SAFETY: this plain C struct is valid when all bytes are zero.
    let algorithm = unsafe { mem::zeroed::<MatmulAlgorithm>() };
    let tf32_key = CublasLtMatmulAlgorithmKey::new(CublasLtMatmulKeyParts {
        data_type: DataType::F32,
        compute_type: ComputeType::F32FastTf32,
        transpose_a: false,
        transpose_b: false,
        epilogue: Epilogue::Default,
        rows: 16,
        inner: 32,
        cols: 8,
        batch_count: 1,
        a_strided_batch_offset: 0,
        b_strided_batch_offset: 0,
        workspace_limit_bytes: 4096,
        library_version: "test-cublaslt".to_string(),
    });
    let f32_key = CublasLtMatmulAlgorithmKey::new(CublasLtMatmulKeyParts {
        data_type: DataType::F32,
        compute_type: ComputeType::F32,
        transpose_a: false,
        transpose_b: false,
        epilogue: Epilogue::Default,
        rows: 16,
        inner: 32,
        cols: 8,
        batch_count: 1,
        a_strided_batch_offset: 0,
        b_strided_batch_offset: 0,
        workspace_limit_bytes: 4096,
        library_version: "test-cublaslt".to_string(),
    });
    {
        let mut supported = runtime.cublaslt_matmul_algorithms().lock().unwrap();
        supported.insert(
            tf32_key.clone(),
            CublasLtMatmulAlgorithmEntry::new(
                algorithm,
                ComputeType::F32FastTf32,
                2048,
                1.0,
                true,
                true,
            ),
        );
        supported.insert(
            f32_key.clone(),
            CublasLtMatmulAlgorithmEntry::new(algorithm, ComputeType::F32, 2048, 2.0, true, true),
        );
    }

    let autotune_key = matmul_autotune_key(device);
    runtime
        .insert_autotune_cache_entry(
            autotune_key.clone(),
            AutotuneCacheEntry::new_cublaslt_matmul_tuning(ComputeType::F32, 2048, None, 0.25),
        )
        .unwrap();

    assert!(
        runtime
            .tuned_cublaslt_matmul_algorithm_for_autotune_key(&autotune_key, &tf32_key)
            .unwrap()
            .is_none()
    );
    let selected = runtime
        .tuned_cublaslt_matmul_algorithm_for_autotune_key(&autotune_key, &f32_key)
        .unwrap()
        .unwrap();

    assert_eq!(selected.compute_type(), ComputeType::F32);
    assert_eq!(selected.workspace_bytes(), 2048);

    cuda::clear_cache(device).unwrap();
}

#[test]
fn test_cuda_backend_tuned_cublaslt_selection_respects_recorded_fingerprint() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let device = Device::Cuda(0);
    let Some(runtime) = skip_if_stub(runtime_for_device(device)) else {
        return;
    };
    cuda::clear_cache(device).unwrap();

    // SAFETY: this plain C struct is valid when all bytes are zero.
    let algorithm = unsafe { mem::zeroed::<MatmulAlgorithm>() };
    let key = CublasLtMatmulAlgorithmKey::new(CublasLtMatmulKeyParts {
        data_type: DataType::F32,
        compute_type: ComputeType::F32FastTf32,
        transpose_a: false,
        transpose_b: false,
        epilogue: Epilogue::Default,
        rows: 16,
        inner: 32,
        cols: 8,
        batch_count: 1,
        a_strided_batch_offset: 0,
        b_strided_batch_offset: 0,
        workspace_limit_bytes: 4096,
        library_version: "test-cublaslt".to_string(),
    });
    let rejected_fingerprint = CublasLtMatmulAlgorithmFingerprint::new(10);
    let selected_fingerprint = CublasLtMatmulAlgorithmFingerprint::new(20);
    {
        let mut supported = runtime.cublaslt_matmul_algorithms().lock().unwrap();
        supported.insert(
            key.clone(),
            CublasLtMatmulAlgorithmEntry::new_with_fingerprint(
                algorithm,
                ComputeType::F32FastTf32,
                2048,
                1.0,
                true,
                true,
                rejected_fingerprint,
            ),
        );
        supported.insert(
            CublasLtMatmulAlgorithmKey::new(CublasLtMatmulKeyParts {
                data_type: DataType::F32,
                compute_type: ComputeType::F32FastTf32,
                transpose_a: false,
                transpose_b: false,
                epilogue: Epilogue::Default,
                rows: 16,
                inner: 32,
                cols: 8,
                batch_count: 1,
                a_strided_batch_offset: 0,
                b_strided_batch_offset: 0,
                workspace_limit_bytes: 2048,
                library_version: "test-cublaslt".to_string(),
            }),
            CublasLtMatmulAlgorithmEntry::new_with_fingerprint(
                algorithm,
                ComputeType::F32FastTf32,
                2048,
                2.0,
                true,
                true,
                selected_fingerprint,
            ),
        );
    }

    let autotune_key = matmul_autotune_key(device);
    runtime
        .insert_autotune_cache_entry(
            autotune_key.clone(),
            AutotuneCacheEntry::new_cublaslt_matmul_tuning(
                ComputeType::F32FastTf32,
                2048,
                Some(selected_fingerprint),
                0.375,
            ),
        )
        .unwrap();

    let selected = runtime
        .tuned_cublaslt_matmul_algorithm_for_autotune_key(&autotune_key, &key)
        .unwrap()
        .unwrap();

    assert_eq!(selected.fingerprint(), Some(selected_fingerprint));
    assert_eq!(selected.workspace_bytes(), 2048);

    cuda::clear_cache(device).unwrap();
}

#[test]
fn test_cuda_backend_selects_tuned_cublaslt_algorithm_from_autotune_cache() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let device = Device::Cuda(0);
    let Some(runtime) = skip_if_stub(runtime_for_device(device)) else {
        return;
    };
    cuda::clear_cache(device).unwrap();

    // SAFETY: this plain C struct is valid when all bytes are zero.
    let algorithm = unsafe { mem::zeroed::<MatmulAlgorithm>() };
    let algorithm_key = CublasLtMatmulAlgorithmKey::new(CublasLtMatmulKeyParts {
        data_type: DataType::F32,
        compute_type: ComputeType::F32FastTf32,
        transpose_a: false,
        transpose_b: false,
        epilogue: Epilogue::Default,
        rows: 16,
        inner: 32,
        cols: 8,
        batch_count: 1,
        a_strided_batch_offset: 0,
        b_strided_batch_offset: 0,
        workspace_limit_bytes: 4096,
        library_version: "test-cublaslt".to_string(),
    });
    {
        let mut supported = runtime.cublaslt_matmul_algorithms().lock().unwrap();
        supported.insert(
            CublasLtMatmulAlgorithmKey::new(CublasLtMatmulKeyParts {
                data_type: DataType::F32,
                compute_type: ComputeType::F32FastTf32,
                transpose_a: false,
                transpose_b: false,
                epilogue: Epilogue::Default,
                rows: 16,
                inner: 32,
                cols: 8,
                batch_count: 1,
                a_strided_batch_offset: 0,
                b_strided_batch_offset: 0,
                workspace_limit_bytes: 0,
                library_version: "test-cublaslt".to_string(),
            }),
            CublasLtMatmulAlgorithmEntry::new(
                algorithm,
                ComputeType::F32FastTf32,
                0,
                1.0,
                true,
                true,
            ),
        );
        supported.insert(
            algorithm_key.clone(),
            CublasLtMatmulAlgorithmEntry::new(
                algorithm,
                ComputeType::F32FastTf32,
                2048,
                3.0,
                true,
                true,
            ),
        );
    }

    let autotune_key = matmul_autotune_key(device);
    runtime
        .insert_autotune_cache_entry(
            autotune_key.clone(),
            AutotuneCacheEntry::new_cublaslt_matmul_tuning(
                ComputeType::F32FastTf32,
                2048,
                None,
                0.125,
            ),
        )
        .unwrap();

    let selected = runtime
        .tuned_cublaslt_matmul_algorithm_for_autotune_key(&autotune_key, &algorithm_key)
        .unwrap()
        .unwrap();

    assert_eq!(selected.workspace_bytes(), 2048);
    assert_eq!(
        runtime
            .autotune_cache()
            .lock()
            .unwrap()
            .get(&autotune_key)
            .unwrap()
            .average_elapsed_ms(),
        Some(0.125)
    );
    let stats = cuda::autotune_cache_stats(device).unwrap();
    assert_eq!(stats.entries(), 1);
    assert_eq!(stats.cublaslt_matmul_tuning_entries(), 1);
    assert_eq!(stats.cublaslt_matmul_tuning_workspace_bytes(), 2048);
    assert_eq!(stats.cublaslt_matmul_tuning_elapsed_ms(), 0.125);
    assert_eq!(
        stats.cublaslt_matmul_tuning_average_elapsed_ms(),
        Some(0.125)
    );
    assert_eq!(
        runtime
            .best_cached_cublaslt_matmul_algorithm_for_key(&algorithm_key, 4096)
            .unwrap()
            .unwrap()
            .workspace_bytes(),
        0
    );

    cuda::clear_cache(device).unwrap();
}
