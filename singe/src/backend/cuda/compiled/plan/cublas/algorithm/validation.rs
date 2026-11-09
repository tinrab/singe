use singe_cublas::{error::Status as CublasStatus, lt::types::Epilogue, types::ComputeType};

use singe_cuda::data_type::DataType as CudaDataType;

use crate::{
    backend::cuda::{
        compiled::plan::cublas::layout::CublasLtMatrixLayout,
        graph_value::CudaGraphValue,
        utility::{ensure_f32, ensure_same_device},
    },
    data_type::DataType,
    error::{Error, Result},
    graph::NumericalPolicy,
    utility::to_usize,
};

#[derive(Debug, Clone, Copy)]
pub struct CublasLtMatmulDTypePlan {
    pub(crate) data_type: DataType,
    pub(crate) cuda_data_type: CudaDataType,
    pub(crate) scale_type: CudaDataType,
    pub(crate) compute_types: &'static [ComputeType],
}

impl CublasLtMatmulDTypePlan {
    pub(crate) fn create(data_type: DataType, epilogue: Epilogue) -> Result<Self> {
        if epilogue != Epilogue::Default && data_type != DataType::F32 {
            return Err(Error::unsupported_operation(
                "cuda graph cuBLASLt non-f32 matmul epilogues are not enabled",
            ));
        }

        let (cuda_data_type, compute_types) = match data_type {
            DataType::F32 if NumericalPolicy::current().precision.allows_tf32() => (
                CudaDataType::F32,
                &[ComputeType::F32FastTf32, ComputeType::F32][..],
            ),
            DataType::F32 => (CudaDataType::F32, &[ComputeType::F32][..]),
            DataType::F16 => (CudaDataType::F16, &[ComputeType::F32][..]),
            DataType::Bf16 => (CudaDataType::Bf16, &[ComputeType::F32][..]),
            _ => {
                return Err(Error::unsupported_data_type("cuda graph cuBLASLt matmul"));
            }
        };
        Ok(Self {
            data_type,
            cuda_data_type,
            scale_type: CudaDataType::F32,
            compute_types,
        })
    }
}

pub fn validate_cublaslt_bias(output: &CudaGraphValue, bias: &CudaGraphValue) -> Result<()> {
    ensure_same_device(bias, output)?;
    ensure_f32(bias)?;
    let output_shape = output.shape().dimensions();
    let bias_shape = bias.shape().dimensions();
    if output_shape.len() != 2 || bias_shape.len() != 1 {
        return Err(Error::unsupported_operation(
            "cuda graph cuBLASLt bias epilogue requires a rank-1 bias for a rank-2 output",
        ));
    }

    if bias_shape[0] != output_shape[1] {
        return Err(Error::shape_mismatch(
            "cuda graph cuBLASLt bias epilogue shape",
            vec![output_shape[1]],
            bias_shape.to_vec(),
        ));
    }

    bias.region()?;
    Ok(())
}

pub fn validate_cublaslt_matmul_problem_dimensions(
    a_layout: &CublasLtMatrixLayout,
    b_layout: &CublasLtMatrixLayout,
) -> Result<()> {
    if a_layout.logical_cols != b_layout.logical_rows {
        return Err(Error::length_mismatch(
            "cublaslt algorithm inner dimension",
            to_usize(
                a_layout.logical_cols,
                "cublaslt algorithm expected inner dimension",
            )?,
            to_usize(
                b_layout.logical_rows,
                "cublaslt algorithm actual inner dimension",
            )?,
        ));
    }
    Ok(())
}

pub fn cublaslt_matmul_heuristic_workspace_fits_limit(
    status: CublasStatus,
    workspace_size: usize,
    workspace_limit_bytes: usize,
) -> bool {
    status == CublasStatus::Success && workspace_size <= workspace_limit_bytes
}
