use crate::{backend::spec::matmul_problem, error::Error, shape::Shape};

#[test]
fn test_matmul_problem_describes_batched_shape() {
    let lhs = Shape::create([2, 3, 4]).unwrap();
    let rhs = Shape::create([2, 4, 5]).unwrap();
    let problem = matmul_problem(&lhs, &rhs, "test matmul").unwrap();
    assert_eq!(problem.batch_dimensions, vec![2]);
    assert_eq!(problem.lhs_rows, 3);
    assert_eq!(problem.inner, 4);
    assert_eq!(problem.rhs_cols, 5);
    assert_eq!(problem.output_shape.dimensions(), &[2, 3, 5]);
    assert!(!problem.is_zero_work());
}

#[test]
fn test_matmul_problem_detects_zero_work() {
    let lhs = Shape::create([2, 0]).unwrap();
    let rhs = Shape::create([0, 3]).unwrap();
    let problem = matmul_problem(&lhs, &rhs, "test matmul").unwrap();
    assert_eq!(problem.output_shape.dimensions(), &[2, 3]);
    assert!(problem.is_zero_work());
}

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
