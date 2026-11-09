use singe::{Device, Shape, Tensor};

fn vector(values: &[f32]) -> Tensor {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create([i64::try_from(values.len()).unwrap()]).unwrap(),
        values.to_vec(),
    )
    .unwrap()
}

#[test]
fn test_operators_chain_tensors_scalars_and_results() {
    let x = vector(&[1., 2., 3.]);
    let weight = vector(&[2., 2., 2.]);
    let bias = vector(&[0.5, 0.5, 0.5]);
    let affine = (&x * &weight + &bias).unwrap();
    assert_eq!(affine.to_vec::<f32>().unwrap(), [2.5, 4.5, 6.5]);
    let shifted = ((2_f32 * &x).unwrap() - 1_f32).unwrap();
    assert_eq!(shifted.to_vec::<f32>().unwrap(), [1., 3., 5.]);
    let reciprocal = (1_f32 / &weight).unwrap();
    assert_eq!(reciprocal.to_vec::<f32>().unwrap(), [0.5, 0.5, 0.5]);
    let negated = (-&x).unwrap();
    assert_eq!(negated.to_vec::<f32>().unwrap(), [-1., -2., -3.]);
    let remainder = (&x % &weight).unwrap();
    assert_eq!(remainder.to_vec::<f32>().unwrap(), [1., 0., 1.]);
}

#[test]
fn test_operator_chains_report_the_first_error() {
    let x = vector(&[1., 2., 3.]);
    let mismatched = vector(&[1., 2.]);
    let result = &x + &mismatched + &x;
    assert!(result.is_err());
}
