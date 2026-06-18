use super::*;

#[test]
fn test_slice_infer_creates_expected_shape() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 5, 6])?,
    ));

    let output = graph.slice_infer(input, &[1..4, 0..3, 2..6])?;

    assert_eq!(graph.shape(output)?.dimensions(), &[3, 3, 4]);
    assert_eq!(graph.shape(output)?.strides(), &[30, 6, 1]);
    assert!(matches!(
        graph.operations.last(),
        Some(Operation::Slice {
            starts,
            limits,
            strides,
            ..
        }) if starts == &[1, 0, 2] && limits == &[4, 3, 6] && strides == &[1, 1, 1]
    ));

    Ok(())
}

#[test]
fn test_strided_slice_infer_creates_expected_shape_and_records_strides() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 5, 6])?,
    ));

    let output = graph.strided_slice_infer(input, &[1..4, 0..5, 0..6], [2, 3, 4])?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 2, 2]);
    assert_eq!(graph.shape(output)?.strides(), &[60, 18, 4]);
    assert!(matches!(
        graph.operations.last(),
        Some(Operation::Slice {
            starts,
            limits,
            strides,
            ..
        }) if starts == &[1, 0, 0] && limits == &[4, 5, 6] && strides == &[2, 3, 4]
    ));

    Ok(())
}

#[test]
fn test_slice_infer_error_removes_generated_output() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2])?.with_strides([i64::MAX])?,
    ));
    let tensor_count = graph.tensors.len();

    let error = graph.slice_infer(input, &[1..2]).unwrap_err();

    assert!(matches!(error, Error::OutOfRange { name } if name == "slice offset"));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_slice_accepts_noncontiguous_output_strides() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 5, 6])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([3, 3, 4])?.with_strides([40, 8, 2])?,
    ));

    graph.slice(input, output, &[1..4, 0..3, 2..6])?;

    Ok(())
}

#[test]
fn test_slice_rejects_mismatched_output_data_type() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 5, 6])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([3, 3, 4])?,
    ));

    let error = graph.slice(input, output, &[1..4, 0..3, 2..6]).unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorDataTypeMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == output
            && operation == "slice output"
            && expected == DataType::F32
            && actual == DataType::F16
    ));

    Ok(())
}

#[test]
fn test_slice_rejects_mismatched_output_dimensions() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([4, 5, 6])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([3, 3, 5])?,
    ));

    let error = graph.slice(input, output, &[1..4, 0..3, 2..6]).unwrap_err();
    assert!(matches!(
        error,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == output
            && operation == "slice output"
            && expected == vec![3, 3, 4]
            && actual == vec![3, 3, 5]
    ));

    Ok(())
}

#[test]
fn test_reshape_rejects_mismatched_element_count() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?));
    let output = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([5])?));

    let err = graph.reshape(input, output).unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorElementCountMismatch {
            tensor_id: _,
            operation,
            expected,
            actual,
        } if operation == "reshape output"
            && expected == 6
            && actual == 5
    ));

    Ok(())
}

#[test]
fn test_reshape_infer_uses_nhwc_default_strides() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 4, 5])?,
    ));

    let output = graph.reshape_infer(input, vec![2, 3, 4, 5])?;

    assert_eq!(graph.shape(output)?.dimensions(), &[2, 3, 4, 5]);
    assert_eq!(graph.shape(output)?.strides(), &[60, 1, 15, 3]);

    Ok(())
}

#[test]
fn test_reshape_infer_error_removes_generated_output() -> Result<()> {
    let mut graph = Graph::new();
    let input = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?));
    let tensor_count = graph.tensors.len();

    let error = graph.reshape_infer(input, vec![5]).unwrap_err();

    assert!(matches!(
        error,
        Error::FrontendTensorElementCountMismatch {
            tensor_id: _,
            operation,
            expected,
            actual,
        } if operation == "reshape output"
            && expected == 6
            && actual == 5
    ));
    assert_eq!(graph.tensors.len(), tensor_count);
    assert!(graph.operations.is_empty());

    Ok(())
}

#[test]
fn test_concat_rejects_mismatched_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let left = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?));
    let right = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?));
    let output = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 5])?));

    let err = graph.concat(&[left, right], output, 1, None).unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == output
            && operation == "concat output shape"
            && expected == vec![2, 6]
            && actual == vec![2, 5]
    ));

    Ok(())
}

#[test]
fn test_concat_rejects_mismatched_input_rank() -> Result<()> {
    let mut graph = Graph::new();
    let left = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?));
    let right = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 3, 1])?,
    ));
    let output = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 6])?));

    let err = graph.concat(&[left, right], output, 1, None).unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == right
            && operation == "concat input rank"
            && expected == vec![2, 3]
            && actual == vec![2, 3, 1]
    ));

    Ok(())
}

#[test]
fn test_concat_rejects_mismatched_non_axis_input_dimension() -> Result<()> {
    let mut graph = Graph::new();
    let left = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?));
    let right = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([4, 5])?));
    let output = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 8])?));

    let err = graph.concat(&[left, right], output, 1, None).unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == right
            && operation == "concat input shape"
            && expected == vec![2, 5]
            && actual == vec![4, 5]
    ));

    Ok(())
}

#[test]
fn test_concat_accepts_noncontiguous_output_strides() -> Result<()> {
    let mut graph = Graph::new();
    let left = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?));
    let right = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?));
    let output = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([2, 6])?.with_strides([16, 2])?,
    ));

    graph.concat(&[left, right], output, 1, None)?;

    assert!(matches!(
        graph.operations.last(),
        Some(Operation::Concat {
            in_place: ConcatInPlaceMode::None,
            ..
        })
    ));

    Ok(())
}

#[test]
fn test_concat_preserves_absent_inplace_index() -> Result<()> {
    let mut graph = Graph::new();
    let left = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?));
    let right = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 3])?));
    let output = graph.tensor(TensorSpec::new(DataType::F32, Shape::contiguous([2, 6])?));

    graph.concat(&[left, right], output, 1, None)?;

    let Some(Operation::Concat { in_place, .. }) = graph.operations.last() else {
        panic!("expected concat operation");
    };
    assert_eq!(*in_place, ConcatInPlaceMode::None);

    graph.concat(&[left, right], output, 1, Some(0))?;

    let Some(Operation::Concat { in_place, .. }) = graph.operations.last() else {
        panic!("expected concat operation");
    };
    assert_eq!(*in_place, ConcatInPlaceMode::Input(0));

    Ok(())
}

#[test]
fn test_paged_cache_load_rejects_mismatched_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([6, 4, 16, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 8, 32])?,
    ));
    let sequence = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let page_table = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));

    let err = graph
        .paged_cache_load(container, output, sequence, page_table)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == output
            && operation == "paged cache load output shape"
            && expected == vec![2, 4, 8, 8]
            && actual == vec![2, 4, 8, 32]
    ));

    Ok(())
}

#[test]
fn test_paged_cache_load_rejects_short_page_table_shape() -> Result<()> {
    let mut graph = Graph::new();
    let container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([6, 4, 16, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 33, 8])?,
    ));
    let sequence = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let page_table = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 2, 1])?,
    ));

    let err = graph
        .paged_cache_load(container, output, sequence, page_table)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::FrontendTensorDimensionsMismatch {
            tensor_id,
            operation,
            expected,
            actual,
        } if tensor_id == page_table
            && operation == "paged cache load page table shape"
            && expected == vec![2, 1, 3, 1]
            && actual == vec![2, 1, 2, 1]
    ));

    Ok(())
}

#[test]
fn test_paged_cache_load_accepts_partial_last_block_output_shape() -> Result<()> {
    let mut graph = Graph::new();
    let container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([6, 4, 16, 8])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 4, 33, 8])?,
    ));
    let sequence = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let page_table = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 3, 1])?,
    ));

    graph.paged_cache_load(container, output, sequence, page_table)?;

    assert!(matches!(
        graph.operations.last(),
        Some(Operation::PagedCacheLoad { .. })
    ));

    Ok(())
}

#[test]
fn test_paged_cache_load_accepts_block_pool_container_shape() -> Result<()> {
    let mut graph = Graph::new();
    let container = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([128, 1, 16, 128])?,
    ));
    let output = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 1, 1024, 128])?,
    ));
    let sequence = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 1, 1])?,
    ));
    let page_table = graph.tensor(TensorSpec::new(
        DataType::I32,
        Shape::contiguous([2, 1, 64, 1])?,
    ));

    graph.paged_cache_load(container, output, sequence, page_table)?;

    assert!(matches!(
        graph.operations.last(),
        Some(Operation::PagedCacheLoad { .. })
    ));

    Ok(())
}
