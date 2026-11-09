use std::mem;

use super::*;

use crate::backend::cuda::runtime::CublasLtMatmulKeyParts;

#[test]
fn test_cuda_backend_selects_best_cached_cublaslt_algorithm_by_key_and_workspace_limit() {
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
                3.0,
                true,
                true,
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
                workspace_limit_bytes: 4096,
                library_version: "test-cublaslt".to_string(),
            }),
            CublasLtMatmulAlgorithmEntry::new(
                algorithm,
                ComputeType::F32FastTf32,
                4096,
                1.5,
                true,
                true,
            ),
        );
        supported.insert(
            CublasLtMatmulAlgorithmKey::new(CublasLtMatmulKeyParts {
                data_type: DataType::F16,
                compute_type: ComputeType::F32FastF16,
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
                ComputeType::F32FastF16,
                0,
                0.5,
                true,
                true,
            ),
        );
    }

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
        workspace_limit_bytes: 0,
        library_version: "test-cublaslt".to_string(),
    });
    let zero_workspace = runtime
        .best_cached_cublaslt_matmul_algorithm_for_key(&key, 0)
        .unwrap()
        .unwrap();
    assert_eq!(zero_workspace.workspace_bytes(), 0);

    let nonzero_workspace = runtime
        .best_cached_cublaslt_matmul_algorithm_for_key(&key, 4096)
        .unwrap()
        .unwrap();
    assert_eq!(nonzero_workspace.workspace_bytes(), 4096);

    cuda::clear_cache(device).unwrap();
}
