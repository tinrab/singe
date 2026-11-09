use std::mem;

use super::*;
use crate::backend::cuda::runtime::{CublasLtMatmulKeyParts, CudaRuntime};

#[test]
fn test_cuda_backend_cublaslt_algorithm_stats_count_supported_workspace_and_waves() {
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
                1.25,
                true,
                false,
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
            CublasLtMatmulAlgorithmEntry::new(algorithm, ComputeType::F32, 2048, 2.5, false, true),
        );
    }

    let stats = cuda::cublaslt_matmul_algorithm_cache_stats(device).unwrap();
    assert_eq!(stats.supported_entries(), 2);
    assert_eq!(stats.supported_zero_workspace_algorithm_entries(), 1);
    assert_eq!(stats.supported_nonzero_workspace_algorithm_entries(), 1);
    assert_eq!(stats.supported_algorithm_workspace_bytes(), 2048);
    assert_eq!(stats.supported_max_algorithm_workspace_bytes(), 2048);
    assert_eq!(stats.supported_algorithm_waves_count(), 3.75);
    assert_eq!(stats.supported_max_algorithm_waves_count(), 2.5);
    assert_eq!(stats.supported_average_algorithm_waves_count(), Some(1.875));
    assert_eq!(stats.supported_nonzero_workspace_limit_entries(), 1);
    assert_eq!(stats.supported_strided_batch_algorithm_entries(), 1);
    assert_eq!(stats.supported_pointer_array_batch_algorithm_entries(), 1);

    cuda::clear_cache(device).unwrap();
}

fn insert_unsupported_algorithm(
    runtime: &CudaRuntime,
    data_type: DataType,
    compute_type: ComputeType,
    lhs_transposed: bool,
    rhs_transposed: bool,
    epilogue: Epilogue,
    workspace_limit: usize,
) {
    runtime
        .unsupported_cublaslt_matmul_algorithms()
        .lock()
        .unwrap()
        .insert(
            CublasLtMatmulAlgorithmKey::new(CublasLtMatmulKeyParts {
                data_type,
                compute_type,
                transpose_a: lhs_transposed,
                transpose_b: rhs_transposed,
                epilogue,
                rows: 16,
                inner: 32,
                cols: 8,
                batch_count: 1,
                a_strided_batch_offset: 0,
                b_strided_batch_offset: 0,
                workspace_limit_bytes: workspace_limit,
                library_version: "test-cublaslt".to_string(),
            }),
            (),
        );
}

fn assert_no_supported_algorithms(stats: &CublasLtMatmulAlgorithmCacheStats) {
    assert_eq!(stats.supported_entries(), 0);
    assert_eq!(stats.supported_zero_workspace_algorithm_entries(), 0);
    assert_eq!(stats.supported_nonzero_workspace_algorithm_entries(), 0);
    assert_eq!(stats.supported_algorithm_workspace_bytes(), 0);
    assert_eq!(stats.supported_max_algorithm_workspace_bytes(), 0);
    assert_eq!(stats.supported_algorithm_waves_count(), 0.0);
    assert_eq!(stats.supported_max_algorithm_waves_count(), 0.0);
    assert_eq!(stats.supported_average_algorithm_waves_count(), None);
    assert_eq!(stats.supported_strided_batch_algorithm_entries(), 0);
    assert_eq!(stats.supported_pointer_array_batch_algorithm_entries(), 0);
    assert_eq!(stats.supported_library_versions(), 0);
    assert_eq!(stats.supported_max_rows(), 0);
    assert_eq!(stats.supported_max_inner(), 0);
    assert_eq!(stats.supported_max_cols(), 0);
    assert_eq!(stats.supported_max_batch_count(), 0);
    assert_eq!(stats.supported_batched_entries(), 0);
    assert_eq!(stats.supported_nonzero_batch_stride_entries(), 0);
    assert_eq!(stats.supported_default_epilogue_entries(), 0);
    assert_eq!(stats.supported_bias_epilogue_entries(), 0);
    assert_eq!(stats.supported_relu_epilogue_entries(), 0);
    assert_eq!(stats.supported_gelu_epilogue_entries(), 0);
    assert_eq!(stats.supported_nonzero_workspace_limit_entries(), 0);
}

#[test]
fn test_cuda_backend_cublaslt_algorithm_stats_count_unsupported_transposed_keys() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let device = Device::Cuda(0);
    let Some(runtime) = skip_if_stub(runtime_for_device(device)) else {
        return;
    };
    cuda::clear_cache(device).unwrap();

    insert_unsupported_algorithm(
        &runtime,
        DataType::F32,
        ComputeType::F32,
        true,
        false,
        Epilogue::Default,
        0,
    );
    insert_unsupported_algorithm(
        &runtime,
        DataType::F16,
        ComputeType::F32FastF16,
        false,
        true,
        Epilogue::Default,
        0,
    );
    insert_unsupported_algorithm(
        &runtime,
        DataType::Bf16,
        ComputeType::F32FastBf16,
        true,
        true,
        Epilogue::Default,
        4096,
    );
    insert_unsupported_algorithm(
        &runtime,
        DataType::F32,
        ComputeType::F32,
        false,
        false,
        Epilogue::Relu,
        0,
    );
    insert_unsupported_algorithm(
        &runtime,
        DataType::F32,
        ComputeType::F32,
        false,
        false,
        Epilogue::Gelu,
        0,
    );
    insert_unsupported_algorithm(
        &runtime,
        DataType::F32,
        ComputeType::F32,
        false,
        false,
        Epilogue::Bias,
        0,
    );

    let stats = cuda::cublaslt_matmul_algorithm_cache_stats(device).unwrap();
    assert_no_supported_algorithms(&stats);
    assert_eq!(stats.unsupported_entries(), 6);
    assert_eq!(stats.unsupported_f32_entries(), 4);
    assert_eq!(stats.unsupported_f16_entries(), 1);
    assert_eq!(stats.unsupported_bf16_entries(), 1);
    assert_eq!(stats.unsupported_f32_compute_entries(), 4);
    assert_eq!(stats.unsupported_tf32_compute_entries(), 0);
    assert_eq!(stats.unsupported_f32_fast_f16_compute_entries(), 1);
    assert_eq!(stats.unsupported_f32_fast_bf16_compute_entries(), 1);
    assert_eq!(stats.unsupported_library_versions(), 1);
    assert_eq!(stats.unsupported_max_rows(), 16);
    assert_eq!(stats.unsupported_max_inner(), 32);
    assert_eq!(stats.unsupported_max_cols(), 8);
    assert_eq!(stats.unsupported_max_batch_count(), 1);
    assert_eq!(stats.unsupported_batched_entries(), 0);
    assert_eq!(stats.unsupported_nonzero_batch_stride_entries(), 0);
    assert_eq!(stats.unsupported_transposed_lhs_entries(), 2);
    assert_eq!(stats.unsupported_transposed_rhs_entries(), 2);
    assert_eq!(stats.unsupported_default_epilogue_entries(), 3);
    assert_eq!(stats.unsupported_bias_epilogue_entries(), 1);
    assert_eq!(stats.unsupported_relu_epilogue_entries(), 1);
    assert_eq!(stats.unsupported_gelu_epilogue_entries(), 1);
    assert_eq!(stats.unsupported_nonzero_workspace_limit_entries(), 1);

    cuda::clear_cache(device).unwrap();
    assert_eq!(
        cuda::cublaslt_matmul_algorithm_cache_stats(device).unwrap(),
        CublasLtMatmulAlgorithmCacheStats::default()
    );
}
