use std::collections::HashMap;

use singe::{DataType, Device, Shape, Tensor, graph::Graph};

#[test]
fn test_fp64_scalar_coefficients_survive_capture_optimization_and_fingerprinting() {
    let coefficient = f64::from_bits(1.0_f64.to_bits() + 1);
    let input =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![1.0_f64, 2.0]).unwrap();
    let changed = input.mul_scalar(2.0).unwrap();
    let capture = |coefficient| {
        Graph::capture([("x", input.clone())], |inputs| {
            Ok(vec![
                inputs[0].mul_scalar(coefficient)?.add_scalar(coefficient)?,
            ])
        })
        .unwrap()
        .0
    };
    let captured = capture(coefficient);
    assert_ne!(
        captured.fingerprint().unwrap(),
        capture(1.0).fingerprint().unwrap()
    );
    let bindings = HashMap::from([("x".into(), changed)]);
    let expected = [
        2.0f64.mul_add(coefficient, coefficient),
        4.0f64.mul_add(coefficient, coefficient),
    ];
    for outputs in [
        captured.run(&bindings).unwrap(),
        captured.compile().unwrap().run(&bindings).unwrap(),
    ] {
        assert_eq!(outputs[0].data_type(), DataType::F64);
        assert_eq!(outputs[0].to_vec::<f64>().unwrap(), expected);
    }
}

#[test]
fn test_i64_scalar_coefficients_preserve_low_bits_and_wrap_on_overflow() {
    let coefficient = (1_i64 << 60) + 1;
    let input = Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![1_i64, 2]).unwrap();
    let (captured, _) = Graph::capture([("x", input.clone())], |inputs| {
        Ok(vec![inputs[0].add_scalar(coefficient)?.sub_scalar(1_i64)?])
    })
    .unwrap();
    let bindings = HashMap::from([("x".into(), input)]);
    for outputs in [
        captured.run(&bindings).unwrap(),
        captured.compile().unwrap().run(&bindings).unwrap(),
    ] {
        assert_eq!(outputs[0].data_type(), DataType::I64);
        assert_eq!(
            outputs[0].to_vec::<i64>().unwrap(),
            [coefficient, coefficient + 1]
        );
    }
    // Integer arithmetic wraps, as `PyTorch`'s does, but a scalar outside `I64` has no `I64` value.
    let maximum = Tensor::scalar(Device::Cpu, i64::MAX).unwrap();
    assert_eq!(
        maximum.add_scalar(1_i64).unwrap().to_vec::<i64>().unwrap(),
        [i64::MIN]
    );
    assert_eq!(
        maximum.mul_scalar(2_i64).unwrap().to_vec::<i64>().unwrap(),
        [-2]
    );
    assert!(maximum.add_scalar(u64::MAX).is_err());
}

#[test]
fn test_weak_scalar_promotion_and_true_division_use_the_tensor_contract() {
    let integers =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![3_i64, 7]).unwrap();
    let weak_float = integers.add_scalar(0.5_f64).unwrap();
    assert_eq!(weak_float.data_type(), DataType::F32);
    assert_eq!(weak_float.to_vec::<f32>().unwrap(), [3.5, 7.5]);
    let strong_float = integers
        .add(&Tensor::scalar(Device::Cpu, 0.5_f64).unwrap())
        .unwrap();
    assert_eq!(strong_float.data_type(), DataType::F64);
    assert_eq!(strong_float.to_vec::<f64>().unwrap(), [3.5, 7.5]);
    let divided = integers.div_scalar(3_i64).unwrap();
    assert_eq!(divided.data_type(), DataType::F32);
    assert_eq!(divided.to_vec::<f32>().unwrap(), [1.0, 7.0_f32 / 3.0]);
    let input = Tensor::scalar(Device::Cpu, 7.0_f32).unwrap();
    assert_ne!((7.0_f32 / 3.0).to_bits(), (7.0_f32 * (1.0 / 3.0)).to_bits());
    let (captured, _) = Graph::capture([("x", input.clone())], |inputs| {
        Ok(vec![inputs[0].div_scalar(3.0)?])
    })
    .unwrap();
    let result = captured
        .compile()
        .unwrap()
        .run(&HashMap::from([("x".into(), input)]))
        .unwrap();
    assert_eq!(
        result[0].to_vec::<f32>().unwrap()[0].to_bits(),
        (7.0_f32 / 3.0).to_bits()
    );
}

#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the test compares direct and double-rounded conversions"
)]
fn test_integer_to_fp32_conversion_does_not_round_through_fp64() {
    // Just above a binary32 midpoint; binary64 would first round back to the midpoint.
    let value = (1_i64 << 62) + (1_i64 << 38) + 1;
    let input = Tensor::scalar(Device::Cpu, value).unwrap();
    assert_ne!((value as f32).to_bits(), ((value as f64) as f32).to_bits());
    assert_eq!(
        input.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap()[0].to_bits(),
        (value as f32).to_bits()
    );
    let zero = Tensor::scalar(Device::Cpu, 0.0_f32).unwrap();
    assert_eq!(
        zero.add_scalar(value).unwrap().to_vec::<f32>().unwrap()[0].to_bits(),
        (value as f32).to_bits()
    );
    assert_eq!(
        zero.add(&input).unwrap().to_vec::<f32>().unwrap()[0].to_bits(),
        (value as f32).to_bits()
    );
}
