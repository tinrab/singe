use crate::{
    backend::spec::{
        concat_problem, embedding_problem, expand_problem, flatten_problem, index_select_problem,
        pad_problem,
    },
    error::Error,
    shape::Shape,
};

#[test]
fn test_concat_problem_describes_output_shape_and_iteration_sizes() {
    let lhs = Shape::create([2, 3, 4]).unwrap();
    let rhs = Shape::create([2, 5, 4]).unwrap();
    let problem = concat_problem(&[&lhs, &rhs], 1, "test concat").unwrap();

    assert_eq!(problem.axis, 1);
    assert_eq!(problem.axis_sizes, vec![3, 5]);
    assert_eq!(problem.output_axis_size, 8);
    assert_eq!(problem.outer_size, 2);
    assert_eq!(problem.inner_size, 4);
    assert_eq!(problem.output_shape.dimensions(), &[2, 8, 4]);
}

#[test]
fn test_concat_problem_normalizes_negative_axis() {
    let lhs = Shape::create([2, 3]).unwrap();
    let rhs = Shape::create([2, 4]).unwrap();
    let problem = concat_problem(&[&lhs, &rhs], -1, "test concat").unwrap();

    assert_eq!(problem.axis, 1);
    assert_eq!(problem.output_shape.dimensions(), &[2, 7]);
}

#[test]
fn test_concat_problem_rejects_non_axis_dimension_mismatch() {
    let lhs = Shape::create([2, 3]).unwrap();
    let rhs = Shape::create([4, 3]).unwrap();

    assert!(matches!(
        concat_problem(&[&lhs, &rhs], 1, "test concat"),
        Err(Error::LengthMismatch { ref name, expected, actual })
            if name == "test concat dimension 0" && expected == 2 && actual == 4
    ));
}

#[test]
fn test_index_select_problem_describes_output_shape_and_iteration_sizes() {
    let input = Shape::create([2, 3, 4]).unwrap();
    let indices = Shape::create([5]).unwrap();
    let problem = index_select_problem(&input, &indices, -2, "test index_select").unwrap();

    assert_eq!(problem.axis, 1);
    assert_eq!(problem.selected_dimension, 3);
    assert_eq!(problem.index_count, 5);
    assert_eq!(problem.outer_size, 2);
    assert_eq!(problem.inner_size, 4);
    assert_eq!(problem.output_shape.dimensions(), &[2, 5, 4]);
}

#[test]
fn test_index_select_problem_rejects_non_vector_indices() {
    let input = Shape::create([2, 3]).unwrap();
    let indices = Shape::create([1, 1]).unwrap();

    assert!(matches!(
        index_select_problem(&input, &indices, 0, "test index_select"),
        Err(Error::InvalidTensorRank {
            operation,
            name,
            expected_rank: 1,
            actual_rank: 2,
            ..
        }) if operation == "test index_select" && name == "indices"
    ));
}

#[test]
fn test_embedding_problem_describes_output_shape() {
    let weight = Shape::create([4, 3]).unwrap();
    let indices = Shape::create([5]).unwrap();
    let problem = embedding_problem(&weight, &indices, "test embedding").unwrap();

    assert_eq!(problem.embedding_count, 4);
    assert_eq!(problem.embedding_dimension, 3);
    assert_eq!(problem.index_count, 5);
    assert_eq!(problem.output_shape.dimensions(), &[5, 3]);
}

#[test]
fn test_embedding_problem_rejects_wrong_weight_rank() {
    let weight = Shape::create([1, 4, 3]).unwrap();
    let indices = Shape::create([5]).unwrap();

    assert!(matches!(
        embedding_problem(&weight, &indices, "test embedding"),
        Err(Error::InvalidTensorRank {
            operation,
            name,
            expected_rank: 2,
            actual_rank: 3,
            ..
        }) if operation == "test embedding" && name == "weight"
    ));
}

#[test]
fn test_pad_problem_describes_output_shape_and_normalized_padding() {
    let input = Shape::create([2, 3, 4]).unwrap();
    let problem = pad_problem(&input, &[(1, 2), (0, 0)], "test pad").unwrap();

    assert_eq!(problem.output_shape.dimensions(), &[5, 3, 4]);
    assert_eq!(problem.padding, vec![(1, 2), (0, 0), (0, 0)]);
    assert!(problem.has_padding);
}

#[test]
fn test_pad_problem_rejects_rank_overflow() {
    let input = Shape::create([2, 3]).unwrap();

    assert!(matches!(
        pad_problem(&input, &[(0, 0), (0, 0), (0, 0)], "test pad"),
        Err(Error::LengthMismatch { ref name, expected, actual })
            if name == "test pad rank" && expected == 2 && actual == 3
    ));
}

#[test]
fn test_pad_problem_crops_with_negative_padding_but_not_below_an_empty_axis() {
    let input = Shape::create([2, 3]).unwrap();

    assert_eq!(
        pad_problem(&input, &[(-1, 0)], "test pad")
            .unwrap()
            .output_shape
            .dimensions(),
        [1, 3]
    );
    assert!(matches!(
        pad_problem(&input, &[(-2, -1)], "test pad"),
        Err(Error::InvalidArgument { ref name }) if name == "test pad axis 0"
    ));
}

#[test]
fn test_flatten_problem_describes_output_shape() {
    let input = Shape::create([2, 3, 4, 5]).unwrap();
    let problem = flatten_problem(&input, 1, 2, "test flatten").unwrap();

    assert_eq!(problem.start_axis, 1);
    assert_eq!(problem.end_axis, 2);
    assert_eq!(problem.flattened_dimension, 12);
    assert_eq!(problem.output_shape.dimensions(), &[2, 12, 5]);
}

#[test]
fn test_flatten_problem_normalizes_negative_axes() {
    let input = Shape::create([2, 3, 4, 5]).unwrap();
    let problem = flatten_problem(&input, -3, -2, "test flatten").unwrap();

    assert_eq!(problem.start_axis, 1);
    assert_eq!(problem.end_axis, 2);
    assert_eq!(problem.output_shape.dimensions(), &[2, 12, 5]);
}

#[test]
fn test_flatten_problem_rejects_reversed_axes() {
    let input = Shape::create([2, 3, 4]).unwrap();

    assert!(matches!(
        flatten_problem(&input, 2, 1, "test flatten"),
        Err(Error::InvalidArgument { ref name }) if name == "test flatten dimension range"
    ));
}

#[test]
fn test_expand_problem_describes_padded_shape_and_strides() {
    let input = Shape::create([3, 1]).unwrap();
    let output = Shape::create([2, 3, 4]).unwrap();
    let problem = expand_problem(&input, &output, "test expand").unwrap();

    assert_eq!(problem.output_shape, output);
    assert_eq!(problem.input_padded_dimensions, vec![1, 3, 1]);
}

#[test]
fn test_expand_problem_rejects_rank_shrink() {
    let input = Shape::create([2, 3]).unwrap();
    let output = Shape::create([3]).unwrap();

    assert!(matches!(
        expand_problem(&input, &output, "test expand"),
        Err(Error::LengthMismatch { ref name, expected, actual })
            if name == "test expand rank" && expected == 1 && actual == 2
    ));
}

#[test]
fn test_expand_problem_rejects_broadcast_mismatch() {
    let input = Shape::create([2, 3]).unwrap();
    let output = Shape::create([2, 4]).unwrap();

    assert!(matches!(
        expand_problem(&input, &output, "test expand"),
        Err(Error::BroadcastMismatch { ref name, .. }) if name == "test expand"
    ));
}
