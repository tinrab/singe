use singe_cublas::{lt::types::Epilogue, types::ComputeType};

use singe_cuda::data_type::DataType as CudaDataType;

use crate::{
    backend::cuda::compiled::plan::cublas::algorithm::CublasLtMatmulDTypePlan, data_type::DataType,
};

#[test]
fn test_cublaslt_dtype_plan_covers_standard_inference_float_matmul_types() {
    let f32_plan = CublasLtMatmulDTypePlan::create(DataType::F32, Epilogue::Default).unwrap();
    assert_eq!(f32_plan.cuda_data_type, CudaDataType::F32);
    assert_eq!(f32_plan.compute_types, &[ComputeType::F32]);

    let f16_plan = CublasLtMatmulDTypePlan::create(DataType::F16, Epilogue::Default).unwrap();
    assert_eq!(f16_plan.cuda_data_type, CudaDataType::F16);
    assert_eq!(f16_plan.compute_types, &[ComputeType::F32]);

    let bf16_plan = CublasLtMatmulDTypePlan::create(DataType::Bf16, Epilogue::Default).unwrap();
    assert_eq!(bf16_plan.cuda_data_type, CudaDataType::Bf16);
    assert_eq!(bf16_plan.compute_types, &[ComputeType::F32]);
}

#[test]
fn test_cublaslt_dtype_plan_keeps_quantized_low_precision_matmul_explicitly_unsupported() {
    assert!(CublasLtMatmulDTypePlan::create(DataType::F8E4M3, Epilogue::Default).is_err());
    assert!(CublasLtMatmulDTypePlan::create(DataType::F8E5M2, Epilogue::Default).is_err());
    assert!(CublasLtMatmulDTypePlan::create(DataType::F4E2M1, Epilogue::Default).is_err());
}

#[test]
fn test_cublaslt_dtype_plan_limits_lower_precision_epilogues_to_future_work() {
    assert!(CublasLtMatmulDTypePlan::create(DataType::F32, Epilogue::Relu).is_ok());
    assert!(CublasLtMatmulDTypePlan::create(DataType::F16, Epilogue::Relu).is_err());
    assert!(CublasLtMatmulDTypePlan::create(DataType::Bf16, Epilogue::Relu).is_err());
}
