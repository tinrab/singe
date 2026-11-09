use singe::{DataType, Device, Tensor};

#[test]
fn test_ranges_preserve_integer_low_bits_and_floating_dtype() {
    let start = (1_i64 << 60) + 1;
    let integers = Tensor::arange(Device::Cpu, start, start + 5, 2_i64).unwrap();
    assert_eq!(integers.data_type(), DataType::I64);
    assert_eq!(
        integers.to_vec::<i64>().unwrap(),
        [start, start + 2, start + 4]
    );
    assert_eq!(
        Tensor::arange(Device::Cpu, i64::MAX, i64::MIN, i64::MIN)
            .unwrap()
            .to_vec::<i64>()
            .unwrap(),
        [i64::MAX, -1]
    );
    assert_eq!(
        Tensor::arange(Device::Cpu, 0.0_f32, 1.0, 0.25)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [0.0, 0.25, 0.5, 0.75]
    );
    assert_eq!(
        Tensor::arange(Device::Cpu, 1.0_f64, 0.0, -0.25)
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        [1.0, 0.75, 0.5, 0.25]
    );
}

#[test]
fn test_floating_ranges_and_linspace_avoid_intermediate_span_overflow() {
    let maximum = f32::MAX;
    assert_eq!(
        Tensor::arange(Device::Cpu, -maximum, maximum, maximum)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [-maximum, 0.0]
    );
    assert_eq!(
        Tensor::linspace(Device::Cpu, -maximum, maximum, 3)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [-maximum, 0.0, maximum]
    );
    assert_eq!(
        Tensor::arange(Device::Cpu, -f64::MAX, f64::MAX, f64::MAX)
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        [-f64::MAX, 0.0]
    );
    assert_eq!(
        Tensor::linspace(Device::Cpu, -f64::MAX, f64::MAX, 3)
            .unwrap()
            .to_vec::<f64>()
            .unwrap(),
        [-f64::MAX, 0.0, f64::MAX]
    );
    let end = f64::from_bits(1.0_f64.to_bits() + 1);
    let values = Tensor::linspace(Device::Cpu, -0.0_f64, end, 7)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    assert_eq!(values[0].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(values[6].to_bits(), end.to_bits());
}

#[test]
fn test_ranges_validate_direction_counts_and_finite_bounds() {
    assert!(Tensor::arange(Device::Cpu, 1_i64, 0, 1).is_err());
    assert!(Tensor::arange(Device::Cpu, 0.0_f64, 1.0, -1.0).is_err());
    assert!(Tensor::arange(Device::Cpu, 0_i64, 1, 0).is_err());
    assert!(Tensor::arange(Device::Cpu, 0.0_f32, f32::INFINITY, 1.0).is_err());
    assert!(Tensor::arange(Device::Cpu, i64::MIN, i64::MAX, 1_i64).is_err());
    assert!(Tensor::linspace(Device::Cpu, 0.0_f64, 1.0, usize::MAX).is_err());
    assert!(Tensor::linspace(Device::Cpu, 0.0_f32, f32::NAN, 0).is_err());
    assert_eq!(
        Tensor::arange(Device::Cpu, 2_i64, 2, -1)
            .unwrap()
            .shape()
            .dimensions(),
        &[0]
    );
    assert_eq!(
        Tensor::linspace(Device::Cpu, 2.0_f64, 3.0, 0)
            .unwrap()
            .shape()
            .dimensions(),
        &[0]
    );
    assert_eq!(
        Tensor::linspace(Device::Cpu, 2.0_f32, 3.0, 1)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        [2.0]
    );
}
