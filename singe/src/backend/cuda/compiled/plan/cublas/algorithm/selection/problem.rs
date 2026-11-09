use singe_cublas::{lt::types::Epilogue, types::ComputeType};

use crate::{
    backend::cuda::{
        compiled::plan::cublas::{
            algorithm::validation::{
                CublasLtMatmulDTypePlan, validate_cublaslt_bias,
                validate_cublaslt_matmul_problem_dimensions,
            },
            layout::{CublasLtMatmulOutputLayout, CublasLtMatrixLayout},
        },
        graph_value::CudaGraphValue,
        runtime::{CublasLtMatmulAlgorithmKey, CublasLtMatmulKeyParts},
    },
    error::{Error, Result},
};

pub struct CublasLtMatmulAlgorithmProblem {
    dtype_plan: CublasLtMatmulDTypePlan,
    output_layout: CublasLtMatmulOutputLayout,
    epilogue: Epilogue,
    rows: u64,
    inner: u64,
    cols: u64,
    batch_count: usize,
    lhs_strided_batch_offset: i64,
    rhs_strided_batch_offset: i64,
    transpose_a: bool,
    transpose_b: bool,
}

impl CublasLtMatmulAlgorithmProblem {
    pub(crate) fn create(
        a_layout: &CublasLtMatrixLayout,
        b_layout: &CublasLtMatrixLayout,
        output: &CudaGraphValue,
        epilogue: Epilogue,
        bias: Option<&CudaGraphValue>,
    ) -> Result<Self> {
        let dtype_plan = CublasLtMatmulDTypePlan::create(output.data_type(), epilogue)?;
        let rows = a_layout.logical_rows;
        let inner = a_layout.logical_cols;
        let cols = b_layout.logical_cols;
        validate_cublaslt_matmul_problem_dimensions(a_layout, b_layout)?;
        if a_layout.batch_count != b_layout.batch_count {
            return Err(Error::unsupported_operation_with_details(
                "cublaslt broadcast batches",
                format!(
                    "lhs batch count {}, rhs batch count {}; use a general matmul provider",
                    a_layout.batch_count, b_layout.batch_count
                ),
            ));
        }

        let output_layout =
            CublasLtMatmulOutputLayout::create(output.layout(), rows, cols, a_layout.batch_count)?;
        if let Some(bias) = bias {
            validate_cublaslt_bias(output, bias)?;
        }

        Ok(Self {
            dtype_plan,
            output_layout,
            epilogue,
            rows,
            inner,
            cols,
            batch_count: a_layout.batch_count,
            lhs_strided_batch_offset: a_layout.strided_batch_offset,
            rhs_strided_batch_offset: b_layout.strided_batch_offset,
            transpose_a: a_layout.is_transposed(),
            transpose_b: b_layout.is_transposed(),
        })
    }

    pub(crate) fn dtype_plan(&self) -> CublasLtMatmulDTypePlan {
        self.dtype_plan
    }

    pub(crate) fn compute_types(&self) -> &[ComputeType] {
        self.dtype_plan.compute_types
    }

    pub(crate) fn output_layout(&self) -> &CublasLtMatmulOutputLayout {
        &self.output_layout
    }

    pub(crate) fn rows(&self) -> u64 {
        self.rows
    }

    pub(crate) fn cols(&self) -> u64 {
        self.cols
    }

    pub(crate) fn key(
        &self,
        compute_type: ComputeType,
        workspace_limit_bytes: usize,
        library_version: String,
    ) -> CublasLtMatmulAlgorithmKey {
        CublasLtMatmulAlgorithmKey::new(CublasLtMatmulKeyParts {
            data_type: self.dtype_plan.data_type,
            compute_type,
            transpose_a: self.transpose_a,
            transpose_b: self.transpose_b,
            epilogue: self.epilogue,
            rows: self.rows,
            inner: self.inner,
            cols: self.cols,
            batch_count: self.batch_count,
            a_strided_batch_offset: self.lhs_strided_batch_offset,
            b_strided_batch_offset: self.rhs_strided_batch_offset,
            workspace_limit_bytes,
            library_version,
        })
    }
}
