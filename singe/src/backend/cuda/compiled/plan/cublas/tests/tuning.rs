use std::{mem, ptr};

use singe_cublas::{lt::matmul::MatmulAlgorithm, types::ComputeType};

use crate::{
    backend::cuda::{
        compiled::plan::cublas::tuning::{
            CublasLtMatmulTuningOptions, CublasLtMatmulTuningResult,
            best_cublaslt_matmul_tuning_result,
        },
        runtime::CublasLtMatmulAlgorithmEntry,
    },
    error::Error,
};

#[test]
fn test_cublaslt_tuning_config_rejects_zero_measurement_iterations() {
    let error = CublasLtMatmulTuningOptions::create(2, 0).unwrap_err();

    assert!(matches!(error, Error::InvalidArgument { ref name }
            if name == "cuBLASLt matmul tuning measurement iterations"));
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
