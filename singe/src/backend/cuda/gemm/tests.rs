use singe_cublas::types::Operation as CublasOperation;

use crate::{backend::cuda::gemm::GemmLayout, layout::Layout};

fn contiguous(dimensions: &[i64]) -> Layout {
    Layout::contiguous(dimensions).unwrap()
}

#[test]
fn test_gemm_folds_a_batched_lhs_times_a_transposed_weight_into_one_gemm() {
    let input = contiguous(&[4, 3, 8]);
    let weight = contiguous(&[16, 8]).transpose(0, 1).unwrap();

    let layout = GemmLayout::create(&input, &weight).unwrap();

    assert_eq!(
        (
            layout.rows,
            layout.columns,
            layout.inner,
            layout.batch_count
        ),
        (12, 16, 8, 1)
    );
    assert_eq!(layout.lhs.operation, CublasOperation::NonTranspose);
    assert_eq!(layout.lhs.leading_dimension, 8);
    assert_eq!(layout.rhs.operation, CublasOperation::Transpose);
    assert_eq!(layout.rhs.leading_dimension, 8);
    assert_eq!(layout.output_shape.dimensions(), [4, 3, 16]);
}

#[test]
fn test_gemm_reads_attention_scores_as_a_strided_batch() {
    let query = contiguous(&[2, 4, 5, 8]);
    let keys = contiguous(&[2, 4, 7, 8]).transpose(2, 3).unwrap();

    let layout = GemmLayout::create(&query, &keys).unwrap();

    assert_eq!(
        (
            layout.rows,
            layout.columns,
            layout.inner,
            layout.batch_count
        ),
        (5, 7, 8, 8)
    );
    assert_eq!(layout.lhs.batch_stride, 40);
    assert_eq!(layout.rhs.operation, CublasOperation::Transpose);
    assert_eq!(layout.rhs.batch_stride, 56);
}

#[test]
fn test_gemm_broadcasts_an_operand_with_a_zero_batch_stride() {
    let lhs = contiguous(&[3, 2, 5, 8]).transpose(2, 3).unwrap();
    let rhs = contiguous(&[1, 1, 5, 6]);

    let layout = GemmLayout::create(&lhs, &rhs).unwrap();

    assert_eq!(layout.batch_count, 6);
    assert_eq!(layout.output_shape.dimensions(), [3, 2, 8, 6]);
    assert_eq!(layout.lhs.operation, CublasOperation::Transpose);
    assert_eq!(layout.rhs.batch_stride, 0);
}

#[test]
fn test_gemm_rejects_matrices_without_a_unit_stride_and_uneven_batches() {
    let strided = contiguous(&[4, 6]).slice(1, 0, 3, 1).unwrap();
    let every_other = contiguous(&[4, 6]).slice(1, 0, 3, 2).unwrap();
    assert!(GemmLayout::create(&every_other, &contiguous(&[3, 5])).is_none());
    assert!(GemmLayout::create(&strided, &contiguous(&[3, 5])).is_some());

    let uneven = contiguous(&[3, 2, 4, 5]).transpose(0, 1).unwrap();
    let rhs = contiguous(&[2, 3, 5, 6]);
    assert!(GemmLayout::create(&uneven, &rhs).is_none());
}

#[test]
fn test_gemm_promotes_vectors() {
    let layout = GemmLayout::create(&contiguous(&[8]), &contiguous(&[8, 3])).unwrap();
    assert_eq!((layout.rows, layout.columns, layout.inner), (1, 3, 8));
    assert_eq!(layout.output_shape.dimensions(), [3]);

    let layout = GemmLayout::create(&contiguous(&[2, 4, 8]), &contiguous(&[8])).unwrap();
    assert_eq!((layout.rows, layout.columns, layout.batch_count), (8, 1, 1));
    assert_eq!(layout.output_shape.dimensions(), [2, 4]);
}
