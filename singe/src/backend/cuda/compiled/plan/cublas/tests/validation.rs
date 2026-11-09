use singe_cublas::error::Status as CublasStatus;

use crate::{
    backend::cuda::compiled::plan::cublas::{
        algorithm::{
            cublaslt_matmul_heuristic_workspace_fits_limit,
            validate_cublaslt_matmul_problem_dimensions,
        },
        launch::validate_cublas_matmul_launch,
        layout::CublasLtMatrixLayout,
        tuning::validate_cublaslt_matmul_tensors,
    },
    device::Device,
    error::Error,
    layout::Layout,
    tensor::Tensor,
};

#[test]
fn test_cublas_matmul_launch_validation_rejects_cpu_tensors_before_shape_checks() {
    let lhs_shape = Layout::contiguous([2, 3]).unwrap();
    let rhs_shape = Layout::contiguous([4, 5]).unwrap();
    let output_shape = Layout::contiguous([2, 5]).unwrap();
    let lhs = Tensor::from_vec(Device::Cpu, lhs_shape.shape(), vec![1.0_f32; 6]).unwrap();
    let rhs = Tensor::from_vec(Device::Cpu, rhs_shape.shape(), vec![1.0_f32; 20]).unwrap();
    let output = Tensor::from_vec(Device::Cpu, output_shape.shape(), vec![0.0_f32; 10]).unwrap();

    let error = validate_cublas_matmul_launch(&lhs, &rhs, &output).unwrap_err();

    assert!(
        matches!(error, Error::DeviceMismatch { ref name, expected, actual }
            if name == "cuda tensor device"
                && expected == Device::Cuda(0)
                && actual == Device::Cpu)
    );
}

#[test]
fn test_cublas_matmul_launch_validation_rejects_cpu_output_before_shape_checks() {
    let lhs_shape = Layout::contiguous([2, 3]).unwrap();
    let rhs_shape = Layout::contiguous([3, 5]).unwrap();
    let output_shape = Layout::contiguous([2, 4]).unwrap();
    let lhs = Tensor::from_vec(Device::Cpu, lhs_shape.shape(), vec![1.0_f32; 6]).unwrap();
    let rhs = Tensor::from_vec(Device::Cpu, rhs_shape.shape(), vec![1.0_f32; 15]).unwrap();
    let output = Tensor::from_vec(Device::Cpu, output_shape.shape(), vec![0.0_f32; 8]).unwrap();

    let error = validate_cublas_matmul_launch(&lhs, &rhs, &output).unwrap_err();

    assert!(
        matches!(error, Error::DeviceMismatch { ref name, expected, actual }
            if name == "cuda tensor device"
                && expected == Device::Cuda(0)
                && actual == Device::Cpu)
    );
}

#[test]
fn test_cublaslt_problem_dimensions_accept_matching_inner_dimension() {
    let lhs_shape = Layout::contiguous([2, 3]).unwrap();
    let rhs_shape = Layout::contiguous([3, 4]).unwrap();
    let lhs = CublasLtMatrixLayout::create_for_layout(&lhs_shape, "lhs").unwrap();
    let rhs = CublasLtMatrixLayout::create_for_layout(&rhs_shape, "rhs").unwrap();

    validate_cublaslt_matmul_problem_dimensions(&lhs, &rhs).unwrap();
}

#[test]
fn test_cublaslt_problem_dimensions_reject_mismatched_inner_dimension() {
    let lhs_shape = Layout::contiguous([2, 3]).unwrap();
    let rhs_shape = Layout::contiguous([4, 5]).unwrap();
    let lhs = CublasLtMatrixLayout::create_for_layout(&lhs_shape, "lhs").unwrap();
    let rhs = CublasLtMatrixLayout::create_for_layout(&rhs_shape, "rhs").unwrap();

    let error = validate_cublaslt_matmul_problem_dimensions(&lhs, &rhs).unwrap_err();

    assert!(
        matches!(error, Error::LengthMismatch { ref name, expected, actual }
            if name == "cublaslt algorithm inner dimension" && expected == 3 && actual == 4)
    );
}

#[test]
fn test_cublaslt_warmup_tensor_validation_rejects_cpu_tensors() {
    let lhs_shape = Layout::contiguous([2, 3]).unwrap();
    let rhs_shape = Layout::contiguous([3, 4]).unwrap();
    let output_shape = Layout::contiguous([2, 4]).unwrap();
    let lhs = Tensor::from_vec(Device::Cpu, lhs_shape.shape(), vec![1.0_f32; 6]).unwrap();
    let rhs = Tensor::from_vec(Device::Cpu, rhs_shape.shape(), vec![1.0_f32; 12]).unwrap();
    let output = Tensor::from_vec(Device::Cpu, output_shape.shape(), vec![1.0_f32; 8]).unwrap();

    let error = validate_cublaslt_matmul_tensors(&lhs, &rhs, &output).unwrap_err();

    assert!(
        matches!(error, Error::DeviceMismatch { ref name, expected, actual }
            if name == "cuda tensor device"
                && expected == Device::Cuda(0)
                && actual == Device::Cpu)
    );
}

#[test]
fn test_cublaslt_matmul_heuristic_workspace_limit_accepts_zero_workspace() {
    assert!(cublaslt_matmul_heuristic_workspace_fits_limit(
        CublasStatus::Success,
        0,
        0
    ));
}

#[test]
fn test_cublaslt_matmul_heuristic_workspace_limit_accepts_nonzero_workspace_within_limit() {
    assert!(cublaslt_matmul_heuristic_workspace_fits_limit(
        CublasStatus::Success,
        4096,
        8192
    ));
}

#[test]
fn test_cublaslt_matmul_heuristic_workspace_limit_rejects_workspace_above_limit() {
    assert!(!cublaslt_matmul_heuristic_workspace_fits_limit(
        CublasStatus::Success,
        8193,
        8192
    ));
}

#[test]
fn test_cublaslt_matmul_heuristic_workspace_limit_rejects_failed_status() {
    assert!(!cublaslt_matmul_heuristic_workspace_fits_limit(
        CublasStatus::NotSupported,
        0,
        8192
    ));
}
