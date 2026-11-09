use crate::{data_type::DataType, error::Error, layout::Layout, shape::Shape};

#[test]
fn test_contiguous_shape() {
    let s = Layout::contiguous(vec![2, 3, 4]).unwrap();
    assert_eq!(s.dimensions(), &[2, 3, 4]);
    assert_eq!(s.strides(), &[12, 4, 1]);
    assert!(s.is_contiguous());
    assert_eq!(s.element_count(), 24);
    assert_eq!(s.storage_span_element_count().unwrap(), 24);
}

#[test]
fn test_storage_span_uses_strided_extent() {
    let shape = Layout::with_strides([2, 2], [100, 1]).unwrap();

    assert_eq!(shape.element_count(), 4);
    assert_eq!(
        shape.shape().byte_size(DataType::F32.size_bytes()).unwrap(),
        16
    );
    assert_eq!(shape.storage_span_element_count().unwrap(), 102);
    assert_eq!(
        shape
            .storage_span_byte_size(DataType::F32.size_bytes())
            .unwrap(),
        408
    );
}

#[test]
fn test_storage_span_is_zero_for_empty_shape() {
    let shape = Layout::with_strides([0, 2], [100, 1]).unwrap();

    assert_eq!(shape.element_count(), 0);
    assert_eq!(shape.storage_span_element_count().unwrap(), 0);
    assert_eq!(
        shape
            .storage_span_byte_size(DataType::F32.size_bytes())
            .unwrap(),
        0
    );
}

#[test]
fn test_broadcast_shapes() {
    let a = Shape::create(vec![2, 3]).unwrap();
    let b = Shape::create(vec![3]).unwrap();
    let out = a.broadcast_with(&b).unwrap();
    assert_eq!(out.dimensions(), &[2, 3]);
}

#[test]
fn test_broadcast_scalar() {
    let a = Shape::create(vec![2, 3]).unwrap();
    let b = Shape::scalar();
    let out = a.broadcast_with(&b).unwrap();
    assert_eq!(out.dimensions(), &[2, 3]);
}

#[test]
fn test_squeeze_preserves_non_contiguous_strides() {
    let shape = Layout::with_strides([2, 1, 3], [3, 99, 1]).unwrap();
    let squeezed = shape.squeeze(1).unwrap();

    assert_eq!(squeezed.dimensions(), &[2, 3]);
    assert_eq!(squeezed.strides(), &[3, 1]);
}

#[test]
fn test_unsqueeze_preserves_view_stride_semantics() {
    let shape = Layout::with_strides([2, 3], [10, 2]).unwrap();

    let front = shape.unsqueeze(0).unwrap();
    assert_eq!(front.dimensions(), &[1, 2, 3]);
    assert_eq!(front.strides(), &[20, 10, 2]);

    let middle = shape.unsqueeze(1).unwrap();
    assert_eq!(middle.dimensions(), &[2, 1, 3]);
    assert_eq!(middle.strides(), &[10, 6, 2]);

    let back = shape.unsqueeze(2).unwrap();
    assert_eq!(back.dimensions(), &[2, 3, 1]);
    assert_eq!(back.strides(), &[10, 2, 1]);
}

#[test]
fn test_reshape_infer_dim() {
    let s = Shape::create(vec![2, 3, 4]).unwrap();
    let reshaped = s.reshape(vec![-1, 4]).unwrap();
    assert_eq!(reshaped.dimensions(), &[6, 4]);
}

#[test]
fn test_reshape_empty_target_produces_rank_zero_scalar() {
    let shape = Shape::create([1]).unwrap();
    let reshaped = shape.reshape(Vec::<i64>::new()).unwrap();
    assert_eq!(reshaped, Shape::scalar());
}

#[test]
fn test_reshape_empty_target_rejects_non_scalar_element_count() {
    let shape = Shape::create([2]).unwrap();
    assert!(matches!(
        shape.reshape(Vec::<i64>::new()),
        Err(Error::ShapeMismatch { name, .. }) if name == "reshape element count"
    ));
}

#[test]
fn test_permute() {
    let s = Shape::create(vec![2, 3, 4]).unwrap();
    let p = s.permute(&[2, 0, 1]).unwrap();
    assert_eq!(p.dimensions(), &[4, 2, 3]);
}

#[test]
fn test_transpose_2d() {
    let s = Shape::create(vec![3, 5]).unwrap();
    let t = s.transpose_2d().unwrap();
    assert_eq!(t.dimensions(), &[5, 3]);
}

#[test]
fn test_permute_rejects_invalid_dimensions_without_panicking() {
    let shape = Shape::create(vec![2, 3, 4]).unwrap();

    assert!(matches!(
        shape.permute(&[0, 1, 3]),
        Err(Error::DimensionOutOfBounds)
    ));
    assert!(matches!(
        shape.permute(&[0, 1, i64::MIN]),
        Err(Error::DimensionOutOfBounds)
    ));
}

#[test]
fn test_permute_rejects_duplicate_dimensions() {
    let shape = Shape::create(vec![2, 3, 4]).unwrap();

    assert!(matches!(
        shape.permute(&[0, 1, 1]),
        Err(Error::InvalidArgument { name }) if name == "permute order"
    ));
}

#[test]
fn test_contiguous_shape_overflowed_stride_is_rejected() {
    assert!(Shape::create([i64::MAX, 2]).is_err());
}

#[test]
fn test_with_strides_rejects_negative_dimensions() {
    assert!(Layout::with_strides(vec![-1], vec![1]).is_err());
}

#[test]
fn test_with_strides_rejects_negative_strides() {
    assert!(Layout::with_strides(vec![2, 3], vec![3, -1]).is_err());
}

#[test]
fn test_with_strides_rejects_overflowing_product() {
    assert!(Layout::with_strides([i64::MAX, 3], [1, 1]).is_err());
}

#[test]
fn test_byte_size_checked_rejects_overflow() {
    let shape = Layout::with_strides([i64::MAX], [0]).unwrap();

    assert!(matches!(
        shape.shape().byte_size(DataType::F64.size_bytes()),
        Err(Error::OutOfRange { name }) if name == "shape byte size"
    ));
}

#[test]
fn test_reshape_overflowed_product_is_rejected() {
    let shape = Shape::create([2]).unwrap();
    assert!(shape.reshape([i64::MAX, i64::MAX]).is_err());
    assert!(shape.reshape([i64::MAX, -1]).is_err());
}

#[test]
fn test_reshape_multiple_inferred_dimensions_rejected() {
    let shape = Shape::create([2, 3]).unwrap();
    assert!(shape.reshape([2, -1, -1]).is_err());
    assert!(shape.reshape([-1, 2, -1]).is_err());
}

#[test]
fn test_reshape_negative_dimension_rejected() {
    let shape = Shape::create([2, 3]).unwrap();
    assert!(shape.reshape([2, -3]).is_err());
}
