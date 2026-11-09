use crate::{
    backend::spec::{reduction_problem, softmax_problem, topk_problem},
    error::Error,
    shape::Shape,
};

#[test]
fn test_reduction_spec_normalizes_negative_dimensions_with_checked_arithmetic() {
    let shape = Shape::create([2, 3, 4]).unwrap();
    let spec = reduction_problem(&shape, &[-1], false).unwrap();
    assert_eq!(spec.output_shape.dimensions(), &[2, 3]);
    let spec = reduction_problem(&shape, &[-2], false).unwrap();
    assert_eq!(spec.output_shape.dimensions(), &[2, 4]);
}

#[test]
fn test_reduction_spec_rejects_negative_dimension_underflow() {
    let shape = Shape::create([2, 3, 4]).unwrap();
    assert!(matches!(
        reduction_problem(&shape, &[i64::MIN], false),
        Err(Error::DimensionOutOfBounds)
    ));
}

#[test]
fn test_reduction_spec_rejects_duplicate_dimensions() {
    let shape = Shape::create([2, 2, 2]).unwrap();
    assert!(
        matches!(reduction_problem(&shape, &[1, 1], true), Err(Error::InvalidArgument { name }) if name == "duplicate reduction axes")
    );
}

#[test]
fn test_softmax_problem_normalizes_axis() {
    let shape = Shape::create([2, 3, 4]).unwrap();
    let problem = softmax_problem(&shape, -2).unwrap();
    assert_eq!(problem.axis, 1);
    assert_eq!(problem.axis_dimension, 3);
    assert_eq!(problem.element_count, 24);
    assert_eq!(problem.output_shape, shape);
}

#[test]
fn test_softmax_problem_detects_empty_tensor() {
    let shape = Shape::create([2, 0, 4]).unwrap();
    let problem = softmax_problem(&shape, 1).unwrap();
    assert_eq!(problem.axis, 1);
    assert_eq!(problem.axis_dimension, 0);
}

#[test]
fn test_softmax_problem_rejects_out_of_range_axis() {
    let shape = Shape::create([2, 3, 4]).unwrap();
    assert!(matches!(
        softmax_problem(&shape, 3),
        Err(Error::DimensionOutOfBounds)
    ));
}

#[test]
fn test_topk_problem_describes_output_shape() {
    let shape = Shape::create([2, 3, 4]).unwrap();
    let problem = topk_problem(&shape, 2, -1, "test topk").unwrap();

    assert_eq!(problem.axis, 2);
    assert_eq!(problem.k, 2);
    assert_eq!(problem.axis_dimension, 4);
    assert_eq!(problem.output_shape.dimensions(), &[2, 3, 2]);
}

#[test]
fn test_topk_problem_rejects_negative_k() {
    let shape = Shape::create([2, 3, 4]).unwrap();

    assert!(matches!(
        topk_problem(&shape, -1, 1, "test topk"),
        Err(Error::InvalidArgument { ref name }) if name == "test topk k"
    ));
}

#[test]
fn test_topk_problem_rejects_k_larger_than_axis() {
    let shape = Shape::create([2, 3, 4]).unwrap();

    assert!(matches!(
        topk_problem(&shape, 5, -1, "test topk"),
        Err(Error::LengthMismatch { ref name, expected, actual })
            if name == "test topk k" && expected == 4 && actual == 5
    ));
}
