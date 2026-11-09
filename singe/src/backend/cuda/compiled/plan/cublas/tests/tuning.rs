use std::{mem, ptr};

use singe_cublas::{
    lt::{matmul::MatmulAlgorithm, types::Epilogue},
    types::ComputeType,
};

#[cfg(feature = "testing")]
use crate::{backend::cuda::runtime::runtime_for_device, testing::skip_if_stub};
use crate::{
    backend::cuda::{
        compiled::plan::cublas::tuning::{
            CublasLtMatmulTuningOptions, CublasLtMatmulTuningResult,
            best_cublaslt_matmul_tuning_result, cublaslt_average_elapsed_ms, cublaslt_workspace,
        },
        runtime::{CublasLtMatmulAlgorithmEntry, CublasLtMatmulAlgorithmKey},
    },
    error::Error,
    storage::cuda::cuda_context_for_device,
    testing::maybe_lock_device,
};
#[cfg(feature = "testing")]
use crate::{
    backend::{
        cuda::{
            self, CudaGraphProvider,
            compiled::plan::cublas::{
                algorithm::with_cublaslt,
                tuning::warm_cublaslt_matmul_algorithm_with_workspace_limit,
            },
            runtime::CublasLtMatmulKeyParts,
            runtime::{AutotuneCacheEntry, AutotuneCacheKey},
        },
        registry::Operation,
    },
    data_type::DataType,
    device::Device,
    shape::Shape,
    tensor::Tensor,
};

#[test]
fn test_cublaslt_tuning_config_reports_default_iteration_counts() {
    assert_eq!(CublasLtMatmulTuningOptions::DEFAULT.warmup_iterations(), 2);
    assert_eq!(
        CublasLtMatmulTuningOptions::DEFAULT.measurement_iterations(),
        5
    );
}

#[test]
fn test_cublaslt_tuning_config_rejects_zero_measurement_iterations() {
    let error = CublasLtMatmulTuningOptions::create(2, 0).unwrap_err();

    assert!(matches!(error, Error::InvalidArgument { ref name }
            if name == "cuBLASLt matmul tuning measurement iterations"));
}

#[test]
fn test_cublaslt_tuning_average_elapsed_time_divides_by_iterations() {
    assert_eq!(cublaslt_average_elapsed_ms(7.5, 3).unwrap(), 2.5);
}

#[test]
fn test_cublaslt_workspace_omits_zero_byte_allocation() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(context) = skip_if_stub(cuda_context_for_device(0)) else {
        return;
    };
    assert!(cublaslt_workspace(&context, 0).unwrap().is_none());
}

#[test]
fn test_cublaslt_tuning_result_selector_prefers_lower_elapsed_time() {
    // SAFETY: this plain C struct is valid when all bytes are zero.
    let algorithm = unsafe { mem::zeroed::<MatmulAlgorithm>() };
    let slower = CublasLtMatmulTuningResult::new(
        CublasLtMatmulAlgorithmEntry::new(algorithm, ComputeType::F32FastTf32, 0, 2.0, true, true),
        0.25,
    );
    let faster = CublasLtMatmulTuningResult::new(
        CublasLtMatmulAlgorithmEntry::new(
            algorithm,
            ComputeType::F32FastTf32,
            4096,
            2.0,
            true,
            true,
        ),
        0.125,
    );

    let selected = best_cublaslt_matmul_tuning_result([&slower, &faster]).unwrap();

    assert!(ptr::eq(selected, &raw const faster));
}

#[test]
fn test_cublaslt_tuning_result_selector_tie_breaks_on_entry_preference() {
    // SAFETY: this plain C struct is valid when all bytes are zero.
    let algorithm = unsafe { mem::zeroed::<MatmulAlgorithm>() };
    let larger_workspace = CublasLtMatmulTuningResult::new(
        CublasLtMatmulAlgorithmEntry::new(
            algorithm,
            ComputeType::F32FastTf32,
            4096,
            2.0,
            true,
            true,
        ),
        0.125,
    );
    let smaller_workspace = CublasLtMatmulTuningResult::new(
        CublasLtMatmulAlgorithmEntry::new(algorithm, ComputeType::F32FastTf32, 0, 2.0, true, true),
        0.125,
    );

    let selected =
        best_cublaslt_matmul_tuning_result([&larger_workspace, &smaller_workspace]).unwrap();

    assert!(ptr::eq(selected, &raw const smaller_workspace));
}

#[cfg(feature = "testing")]
#[test]
fn test_cublaslt_warmup_prefers_tuned_fp32_winner() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let device = Device::Cuda(0);
    let Some(runtime) = skip_if_stub(runtime_for_device(device)) else {
        return;
    };
    cuda::clear_cache(device).unwrap();

    let library_version =
        with_cublaslt(&runtime, |cublaslt| Ok(cublaslt.version()?.to_string())).unwrap();
    // SAFETY: this plain C struct is valid when all bytes are zero.
    let algorithm = unsafe { mem::zeroed::<MatmulAlgorithm>() };
    let tuned_key = CublasLtMatmulAlgorithmKey::new(CublasLtMatmulKeyParts {
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
        library_version: library_version.clone(),
    });
    {
        let mut algorithms = runtime.cublaslt_matmul_algorithms().lock().unwrap();
        algorithms.insert(
            CublasLtMatmulAlgorithmKey::new(CublasLtMatmulKeyParts {
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
                workspace_limit_bytes: 0,
                library_version: library_version.clone(),
            }),
            CublasLtMatmulAlgorithmEntry::new(algorithm, ComputeType::F32, 0, 1.0, true, true),
        );
        algorithms.insert(
            tuned_key,
            CublasLtMatmulAlgorithmEntry::new(algorithm, ComputeType::F32, 2048, 4.0, true, true),
        );
    }

    let lhs_shape = Shape::create([16, 32]).unwrap();
    let lhs = Tensor::from_vec(device, &lhs_shape, vec![0.0_f32; 16 * 32]).unwrap();
    let rhs_shape = Shape::create([32, 8]).unwrap();
    let rhs = Tensor::from_vec(device, &rhs_shape, vec![0.0_f32; 32 * 8]).unwrap();
    let output_shape = Shape::create([16, 8]).unwrap();
    let output = Tensor::from_vec(device, &output_shape, vec![0.0_f32; 16 * 8]).unwrap();
    let autotune_key = AutotuneCacheKey::from_tensors(
        Operation::MatMul,
        DataType::F32,
        [&lhs, &rhs, &output],
        CudaGraphProvider::CublasLt,
        library_version,
        Some(4096),
    );
    runtime
        .insert_autotune_cache_entry(
            autotune_key,
            AutotuneCacheEntry::new_cublaslt_matmul_tuning(ComputeType::F32, 2048, None, 0.125),
        )
        .unwrap();

    let selected = warm_cublaslt_matmul_algorithm_with_workspace_limit(
        &runtime,
        &lhs,
        &rhs,
        &output,
        Epilogue::Default,
        None,
        4096,
    )
    .unwrap()
    .unwrap();

    assert_eq!(selected.workspace_bytes(), 2048);
    cuda::clear_cache(device).unwrap();
}
