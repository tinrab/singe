use singe_cublas::{lt::descriptor::MatrixLayout, types::Operation as CublasOperation};

use crate::{
    backend::cuda::tensor_region::CudaTensorSource,
    error::{Error, Result},
    layout::Layout,
    utility::{
        checked_element_count, checked_mul, checked_mul_i64, checked_mul_u64, to_i64, to_u64,
        to_usize,
    },
};

pub fn cublaslt_batch_count(batch_dimensions: &[i64]) -> Result<usize> {
    checked_element_count(batch_dimensions, "cuda graph cuBLASLt batch count")
}

pub fn cublaslt_batch_prefix_is_contiguous(
    batch_dimensions: &[i64],
    batch_strides: &[i64],
    matrix_element_count: i64,
) -> Result<bool> {
    if batch_dimensions.is_empty() {
        return Ok(true);
    }

    let mut expected_stride = matrix_element_count;
    for index in (0..batch_dimensions.len()).rev() {
        if batch_strides[index] != expected_stride {
            return Ok(false);
        }

        expected_stride = checked_mul_i64(
            expected_stride,
            batch_dimensions[index],
            "cuda graph cuBLASLt batch stride",
        )?;
    }
    Ok(true)
}

#[derive(Debug, Clone, Copy)]
pub struct CublasLtMatrixLayout {
    pub(crate) operation: CublasOperation,
    pub(crate) descriptor_rows: u64,
    pub(crate) descriptor_cols: u64,
    pub(crate) leading_dimension: i64,
    pub(crate) batch_count: usize,
    pub(crate) strided_batch_offset: i64,
    pub(crate) logical_rows: u64,
    pub(crate) logical_cols: u64,
    pub(crate) element_count: usize,
}

impl CublasLtMatrixLayout {
    pub(crate) fn new(
        operation: CublasOperation,
        descriptor_rows: u64,
        descriptor_cols: u64,
        leading_dimension: i64,
        batch_count: usize,
        strided_batch_offset: i64,
        logical_rows: u64,
        logical_cols: u64,
        element_count: usize,
    ) -> Self {
        Self {
            operation,
            descriptor_rows,
            descriptor_cols,
            leading_dimension,
            batch_count,
            strided_batch_offset,
            logical_rows,
            logical_cols,
            element_count,
        }
    }

    pub(crate) fn is_transposed(self) -> bool {
        self.operation == CublasOperation::Transpose
    }

    pub(crate) fn configure_descriptor(self, descriptor: &mut MatrixLayout) -> Result<()> {
        if self.batch_count > 1 {
            descriptor.set_strided_batch(self.batch_count, self.strided_batch_offset)?;
        }
        Ok(())
    }
}

impl CublasLtMatrixLayout {
    pub(crate) fn create(tensor: &(impl CudaTensorSource + ?Sized), name: &str) -> Result<Self> {
        Self::create_for_layout(tensor.layout(), name)
    }

    pub(crate) fn create_for_layout(shape: &Layout, name: &str) -> Result<Self> {
        let dimensions = shape.dimensions();
        let strides = shape.strides();
        let (&[ref batch @ .., matrix_rows, matrix_columns], true) =
            (dimensions, dimensions.len() == strides.len())
        else {
            return Err(Error::unsupported_operation(format!(
                "cuda graph cuBLASLt {name} only supports rank-2 matrices or strided batches"
            )));
        };

        // An offset needs nothing here: operands' regions start at it (`CudaGraphValue::region`).
        let (batch_strides, matrix_strides) = strides.split_at(batch.len());
        let batch_count = cublaslt_batch_count(batch)?;
        let rows = to_u64(matrix_rows, "cuda graph cuBLASLt matrix rows")?;
        let cols = to_u64(matrix_columns, "cuda graph cuBLASLt matrix columns")?;
        let matrix_element_count =
            checked_mul_u64(rows, cols, "cuda graph cuBLASLt matrix element count")?;
        let element_count = to_usize(
            matrix_element_count,
            "cuda graph cuBLASLt matrix element count",
        )?;
        let element_count = checked_mul(
            element_count,
            batch_count,
            "cuda graph cuBLASLt strided batch element count",
        )?;
        let expected_batch_stride = to_i64(
            matrix_element_count,
            "cuda graph cuBLASLt matrix batch stride",
        )?;
        let strided_batch_offset = if batch_count > 1 {
            expected_batch_stride
        } else {
            0
        };
        let is_valid_batch_stride =
            cublaslt_batch_prefix_is_contiguous(batch, batch_strides, expected_batch_stride)?;
        if is_valid_batch_stride && matrix_strides == [matrix_columns, 1] {
            Ok(Self::new(
                CublasOperation::NonTranspose,
                rows,
                cols,
                matrix_columns,
                batch_count,
                strided_batch_offset,
                rows,
                cols,
                element_count,
            ))
        } else if is_valid_batch_stride && matrix_strides == [1, matrix_rows] {
            Ok(Self::new(
                CublasOperation::Transpose,
                cols,
                rows,
                matrix_rows,
                batch_count,
                strided_batch_offset,
                rows,
                cols,
                element_count,
            ))
        } else {
            let details = if batch.is_empty() {
                format!(
                    "cuda graph cuBLASLt {name} supports only contiguous or transposed-contiguous rank-2 layouts"
                )
            } else {
                format!(
                    "cuda graph cuBLASLt {name} supports only contiguous or transposed-contiguous strided-batch layouts"
                )
            };
            Err(Error::unsupported_operation(details))
        }
    }
}

#[derive(Debug, Clone)]
pub struct CublasLtMatmulOutputLayout {
    pub(crate) batch_count: usize,
    pub(crate) strided_batch_offset: i64,
    pub(crate) element_count: usize,
}

impl CublasLtMatmulOutputLayout {
    pub(crate) fn create(shape: &Layout, rows: u64, cols: u64, batch_count: usize) -> Result<Self> {
        let rows_i64 = to_i64(rows, "cuda graph cuBLASLt output rows")?;
        let cols_i64 = to_i64(cols, "cuda graph cuBLASLt output columns")?;
        let dimensions = shape.dimensions();
        let strides = shape.strides();
        let (&[ref batch @ .., matrix_rows, matrix_columns], true) =
            (dimensions, dimensions.len() == strides.len())
        else {
            return Err(Error::unsupported_operation(
                "cuda graph cuBLASLt output supports only rank-2 matrices or strided batches",
            ));
        };

        let (batch_strides, matrix_strides) = strides.split_at(batch.len());
        let actual_batch_count = cublaslt_batch_count(batch)?;
        if actual_batch_count != batch_count {
            return Err(Error::length_mismatch(
                "cublaslt matmul output batch count",
                batch_count,
                actual_batch_count,
            ));
        }

        if matrix_rows != rows_i64 || matrix_columns != cols_i64 {
            let mut expected_dimensions = batch.to_vec();
            expected_dimensions.push(rows_i64);
            expected_dimensions.push(cols_i64);
            return Err(Error::shape_mismatch(
                "cublaslt matmul output shape",
                expected_dimensions,
                dimensions.to_vec(),
            ));
        }

        let matrix_element_count = checked_mul_u64(
            rows,
            cols,
            "cuda graph cuBLASLt output matrix element count",
        )?;
        let strided_batch_offset = if batch_count > 1 {
            let expected_stride = to_i64(
                matrix_element_count,
                "cuda graph cuBLASLt output batch stride",
            )?;
            let batch_prefix_is_contiguous =
                cublaslt_batch_prefix_is_contiguous(batch, batch_strides, expected_stride)?;
            if !batch_prefix_is_contiguous || matrix_strides != [cols_i64, 1] {
                return Err(Error::unsupported_operation(
                    "cuda graph cuBLASLt output supports only contiguous strided-batch layouts",
                ));
            }
            expected_stride
        } else {
            if matrix_strides != [cols_i64, 1] {
                return Err(Error::unsupported_operation(
                    "cuda graph cuBLASLt output supports only contiguous rank-2 layouts",
                ));
            }
            0
        };
        let element_count = to_usize(
            matrix_element_count,
            "cuda graph cuBLASLt output matrix element count",
        )?;
        let element_count = checked_mul(
            element_count,
            batch_count,
            "cuda graph cuBLASLt output element count",
        )?;
        Ok(Self {
            batch_count,
            strided_batch_offset,
            element_count,
        })
    }

    pub(crate) fn configure_descriptor(&self, descriptor: &mut MatrixLayout) -> Result<()> {
        if self.batch_count > 1 {
            descriptor.set_strided_batch(self.batch_count, self.strided_batch_offset)?;
        }
        Ok(())
    }
}
