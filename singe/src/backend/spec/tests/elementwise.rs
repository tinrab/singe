use crate::{
    backend::spec::{clamp_problem, where_elementwise_problem},
    error::Error,
    shape::Shape,
};

#[test]
fn test_clamp_problem_rejects_inverted_bounds() {
    let shape = Shape::create([2, 3]).unwrap();

    assert!(matches!(
        clamp_problem(&shape, 2.0, 1.0),
        Err(Error::InvalidArgument { ref name }) if name == "clamp bounds"
    ));
}

#[test]
fn test_where_elementwise_problem_describes_three_way_broadcast() {
    let condition = Shape::create([2, 1, 4]).unwrap();
    let lhs = Shape::create([1, 3, 4]).unwrap();
    let rhs = Shape::create([2, 3, 1]).unwrap();
    let problem = where_elementwise_problem(&condition, &lhs, &rhs).unwrap();

    assert_eq!(problem.output_shape.dimensions(), &[2, 3, 4]);
    assert_eq!(
        problem.input_padded_dimensions,
        vec![vec![2, 1, 4], vec![1, 3, 4], vec![2, 3, 1]]
    );
}

#[test]
fn test_where_elementwise_problem_rejects_broadcast_mismatch() {
    let condition = Shape::create([2, 1]).unwrap();
    let lhs = Shape::create([2, 3]).unwrap();
    let rhs = Shape::create([4, 3]).unwrap();

    assert!(matches!(
        where_elementwise_problem(&condition, &lhs, &rhs),
        Err(Error::BroadcastMismatch { .. })
    ));
}
