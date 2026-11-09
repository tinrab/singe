use singe_cublas::types::Operation as CublasOperation;

use crate::{
    backend::cuda::compiled::plan::cublas::layout::{
        CublasLtMatmulOutputLayout, CublasLtMatrixLayout,
    },
    error::Error,
    layout::Layout,
};

#[test]
fn test_cublaslt_matrix_layout_parses_rank3_strided_batch() {
    let shape = Layout::contiguous([2, 3, 4]).unwrap();

    let layout = CublasLtMatrixLayout::create_for_layout(&shape, "lhs").unwrap();

    assert_eq!(layout.operation, CublasOperation::NonTranspose);
    assert_eq!(layout.descriptor_rows, 3);
    assert_eq!(layout.descriptor_cols, 4);
    assert_eq!(layout.leading_dimension, 4);
    assert_eq!(layout.batch_count, 2);
    assert_eq!(layout.strided_batch_offset, 12);
    assert_eq!(layout.logical_rows, 3);
    assert_eq!(layout.logical_cols, 4);
    assert_eq!(layout.element_count, 24);
}

#[test]
fn test_cublaslt_matrix_layout_parses_rank3_transposed_strided_batch() {
    let shape = Layout::with_strides([2, 3, 4], [12, 1, 3]).unwrap();

    let layout = CublasLtMatrixLayout::create_for_layout(&shape, "lhs").unwrap();

    assert_eq!(layout.operation, CublasOperation::Transpose);
    assert_eq!(layout.descriptor_rows, 4);
    assert_eq!(layout.descriptor_cols, 3);
    assert_eq!(layout.leading_dimension, 3);
    assert_eq!(layout.batch_count, 2);
    assert_eq!(layout.strided_batch_offset, 12);
    assert_eq!(layout.logical_rows, 3);
    assert_eq!(layout.logical_cols, 4);
    assert_eq!(layout.element_count, 24);
}

#[test]
fn test_cublaslt_matrix_layout_rejects_rank3_non_strided_batch() {
    let shape = Layout::with_strides([2, 3, 4], [16, 4, 1]).unwrap();

    let error = CublasLtMatrixLayout::create_for_layout(&shape, "lhs").unwrap_err();

    assert!(
        matches!(error, Error::Unsupported { ref operation, .. } if operation.contains("strided-batch"))
    );
}

#[test]
fn test_cublaslt_matrix_layout_rejects_rank4_non_contiguous_batch_prefix() {
    let shape = Layout::with_strides([2, 5, 3, 4], [72, 12, 4, 1]).unwrap();

    let error = CublasLtMatrixLayout::create_for_layout(&shape, "lhs").unwrap_err();

    assert!(
        matches!(error, Error::Unsupported { ref operation, .. } if operation.contains("strided-batch"))
    );
}

#[test]
fn test_cublaslt_matrix_layout_parses_rank4_strided_batch() {
    let shape = Layout::contiguous([2, 5, 3, 4]).unwrap();

    let layout = CublasLtMatrixLayout::create_for_layout(&shape, "lhs").unwrap();

    assert_eq!(layout.operation, CublasOperation::NonTranspose);
    assert_eq!(layout.descriptor_rows, 3);
    assert_eq!(layout.descriptor_cols, 4);
    assert_eq!(layout.leading_dimension, 4);
    assert_eq!(layout.batch_count, 10);
    assert_eq!(layout.strided_batch_offset, 12);
    assert_eq!(layout.logical_rows, 3);
    assert_eq!(layout.logical_cols, 4);
    assert_eq!(layout.element_count, 120);
}

#[test]
fn test_cublaslt_matrix_layout_parses_rank4_transposed_strided_batch() {
    let shape = Layout::with_strides([2, 5, 3, 4], [60, 12, 1, 3]).unwrap();

    let layout = CublasLtMatrixLayout::create_for_layout(&shape, "lhs").unwrap();

    assert_eq!(layout.operation, CublasOperation::Transpose);
    assert_eq!(layout.descriptor_rows, 4);
    assert_eq!(layout.descriptor_cols, 3);
    assert_eq!(layout.leading_dimension, 3);
    assert_eq!(layout.batch_count, 10);
    assert_eq!(layout.strided_batch_offset, 12);
    assert_eq!(layout.logical_rows, 3);
    assert_eq!(layout.logical_cols, 4);
    assert_eq!(layout.element_count, 120);
}

#[test]
fn test_cublaslt_output_layout_parses_rank2_output() {
    let shape = Layout::contiguous([3, 4]).unwrap();

    let layout = CublasLtMatmulOutputLayout::create(&shape, 3, 4, 1).unwrap();

    assert_eq!(layout.batch_count, 1);
    assert_eq!(layout.strided_batch_offset, 0);
    assert_eq!(layout.element_count, 12);
}

#[test]
fn test_cublaslt_output_layout_parses_rank3_strided_batch_output() {
    let shape = Layout::contiguous([2, 3, 4]).unwrap();

    let layout = CublasLtMatmulOutputLayout::create(&shape, 3, 4, 2).unwrap();

    assert_eq!(layout.batch_count, 2);
    assert_eq!(layout.strided_batch_offset, 12);
    assert_eq!(layout.element_count, 24);
}

#[test]
fn test_cublaslt_output_layout_parses_rank4_strided_batch_output() {
    let shape = Layout::contiguous([2, 5, 3, 4]).unwrap();

    let layout = CublasLtMatmulOutputLayout::create(&shape, 3, 4, 10).unwrap();

    assert_eq!(layout.batch_count, 10);
    assert_eq!(layout.strided_batch_offset, 12);
    assert_eq!(layout.element_count, 120);
}

#[test]
fn test_cublaslt_output_layout_rejects_rank3_non_contiguous_batch_output() {
    let shape = Layout::with_strides([2, 3, 4], [16, 4, 1]).unwrap();

    let error = CublasLtMatmulOutputLayout::create(&shape, 3, 4, 2).unwrap_err();

    assert!(
        matches!(error, Error::Unsupported { ref operation, .. } if operation.contains("strided-batch"))
    );
}

#[test]
fn test_cublaslt_output_layout_rejects_rank4_non_contiguous_batch_prefix() {
    let shape = Layout::with_strides([2, 5, 3, 4], [72, 12, 4, 1]).unwrap();

    let error = CublasLtMatmulOutputLayout::create(&shape, 3, 4, 10).unwrap_err();

    assert!(
        matches!(error, Error::Unsupported { ref operation, .. } if operation.contains("strided-batch"))
    );
}
