use crate::{backend::spec::matmul_problem, error::Error, shape::Shape};

#[test]
fn test_matmul_problem_rejects_mismatched_inner_dimension() {
    let lhs = Shape::create([2, 3]).unwrap();
    let rhs = Shape::create([4, 5]).unwrap();
    assert!(matches!(
        matmul_problem(&lhs, &rhs, "test matmul"),
        Err(Error::LengthMismatch { name, expected: 3, actual: 4 })
            if name == "test matmul inner dimension"
    ));
}

#[test]
fn test_matmul_problem_rejects_mismatched_batch_dimension() {
    let lhs = Shape::create([2, 3, 4]).unwrap();
    let rhs = Shape::create([5, 4, 6]).unwrap();
    assert!(matches!(
        matmul_problem(&lhs, &rhs, "test matmul"),
        Err(Error::BroadcastMismatch { name, left, right })
            if name == "test matmul batch dimensions" && left == [2] && right == [5]
    ));
}
