use std::collections::HashMap;

use singe::{
    DataType, Device, Shape, Tensor,
    error::Error,
    graph::{Graph, capture},
};

#[test]
fn test_quantization_rejects_non_positive_and_non_finite_scales() {
    let input = Tensor::from_vec(Device::Cpu, Shape::create([1]).unwrap(), vec![1_f32]).unwrap();
    for dtype in [DataType::I8, DataType::U8] {
        for scale in [0., -0., -1., f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                input.quantize(scale, 0, dtype),
                Err(Error::InvalidArgument { .. })
            ));
        }
    }
}

#[test]
fn test_quantization_non_finite_inputs_have_explicit_saturating_results() {
    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([3]).unwrap(),
        vec![f32::NAN, f32::NEG_INFINITY, f32::INFINITY],
    )
    .unwrap();
    assert_eq!(
        input
            .quantize(1., 3, DataType::I8)
            .unwrap()
            .to_vec::<i8>()
            .unwrap(),
        [3, -128, 127]
    );
    assert_eq!(
        input
            .quantize(1., 3, DataType::U8)
            .unwrap()
            .to_vec::<u8>()
            .unwrap(),
        [3, 0, 255]
    );
}

#[test]
fn test_folded_quantization_rounds_halfway_values_to_even() {
    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([6]).unwrap(),
        vec![-2.5_f32, -1.5, -0.5, 0.5, 1.5, 2.5],
    )
    .unwrap();
    let (captured, _) = Graph::capture(Vec::<(String, Tensor)>::new(), |_| {
        let input = capture::constant("halfway", &input)?;
        Ok(vec![input.quantize(1., 0, DataType::I8)?])
    })
    .unwrap();
    let compiled = captured.compile().unwrap();
    assert_eq!(
        compiled.run(&HashMap::new()).unwrap()[0]
            .to_vec::<i8>()
            .unwrap(),
        [-2, -2, 0, 0, 2, 2]
    );
}

#[test]
fn test_per_channel_parameters_apply_along_their_axis() {
    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([2, 2]).unwrap(),
        vec![1_f32, 2., 3., 4.],
    )
    .unwrap();
    let scale =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![0.5_f32, 1.]).unwrap();
    let zero_point =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![0_i64, 10]).unwrap();
    let quantized = input
        .quantize_with(&scale, &zero_point, Some(1), None, DataType::I8)
        .unwrap();
    assert_eq!(quantized.to_vec::<i8>().unwrap(), [2, 12, 6, 14]);
    let restored = quantized
        .dequantize_with(&scale, &zero_point, Some(1), None)
        .unwrap();
    assert_eq!(restored.to_vec::<f32>().unwrap(), [1., 2., 3., 4.]);
    let wrong = Tensor::from_vec(Device::Cpu, Shape::create([3]).unwrap(), vec![1_f32; 3]).unwrap();
    assert!(
        input
            .quantize_with(&wrong, &zero_point, Some(1), None, DataType::I8)
            .is_err()
    );
}

fn matrix(dimensions: &[i64], values: Vec<f32>) -> Tensor {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create(dimensions.to_vec()).unwrap(),
        values,
    )
    .unwrap()
}

fn block_parameters() -> (Tensor, Tensor) {
    let scale = matrix(&[2, 2], vec![0.5, 1., 2., 4.]);
    let zero_point = Tensor::from_vec(
        Device::Cpu,
        Shape::create([2, 2]).unwrap(),
        vec![0_i64, 1, 2, 3],
    )
    .unwrap();
    (scale, zero_point)
}

#[test]
fn test_block_scaled_parameters_apply_to_consecutive_positions_along_their_axis() {
    let input = matrix(&[2, 4], (1..=8).map(|value| value as f32).collect());
    let (scale, zero_point) = block_parameters();
    let quantized = input
        .quantize_with(&scale, &zero_point, Some(1), Some(2), DataType::I8)
        .unwrap();
    assert_eq!(quantized.to_vec::<i8>().unwrap(), [2, 4, 4, 5, 4, 5, 5, 5]);
    let restored = quantized
        .dequantize_with(&scale, &zero_point, Some(1), Some(2))
        .unwrap();
    assert_eq!(
        restored.to_vec::<f32>().unwrap(),
        [1., 2., 3., 4., 4., 6., 8., 8.]
    );
    assert!(
        input
            .quantize_with(&scale, &zero_point, Some(1), Some(3), DataType::I8)
            .is_err()
    );
    assert!(
        input
            .quantize_with(&scale, &zero_point, None, Some(2), DataType::I8)
            .is_err()
    );
}

#[cfg(all(feature = "cuda", feature = "testing"))]
#[test]
fn test_cuda_quantization_reads_parameters_on_the_device_and_matches_cpu() {
    use singe::testing::{maybe_lock_device, skip_if_stub};

    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let cuda = Device::Cuda(0);
    let input = matrix(&[2, 4], vec![1., -2.5, 3.25, f32::NAN, 5., 6.5, -7., 1000.]);
    let Some(device_input) = skip_if_stub(input.to_device(cuda)) else {
        return;
    };
    let (block_scale, block_zero_point) = block_parameters();
    let channel_scale = matrix(&[4], vec![0.5, 1., 2., 4.]);
    let channel_zero_point = Tensor::from_vec(
        Device::Cpu,
        Shape::create([4]).unwrap(),
        vec![0_i64, 1, -2, 3],
    )
    .unwrap();
    let tensor_scale = Tensor::scalar(Device::Cpu, 0.25_f32).unwrap();
    let tensor_zero_point = Tensor::scalar(Device::Cpu, 5_i64).unwrap();
    let cases = [
        (&tensor_scale, &tensor_zero_point, None, None),
        (&channel_scale, &channel_zero_point, Some(1), None),
        (&block_scale, &block_zero_point, Some(1), Some(2)),
    ];
    for (scale, zero_point, axis, block_size) in cases {
        let device_scale = scale.to_device(cuda).unwrap();
        let device_zero_point = zero_point.to_device(cuda).unwrap();
        for data_type in [DataType::I8, DataType::U8] {
            let expected = input
                .quantize_with(scale, zero_point, axis, block_size, data_type)
                .unwrap();
            let actual = device_input
                .quantize_with(
                    &device_scale,
                    &device_zero_point,
                    axis,
                    block_size,
                    data_type,
                )
                .unwrap();
            assert_eq!(actual.to_bytes().unwrap(), expected.to_bytes().unwrap());
            let restored = expected
                .dequantize_with(scale, zero_point, axis, block_size)
                .unwrap();
            let device_restored = actual
                .dequantize_with(&device_scale, &device_zero_point, axis, block_size)
                .unwrap();
            assert_eq!(
                device_restored.to_vec::<f32>().unwrap(),
                restored.to_vec::<f32>().unwrap()
            );
        }
    }
    let invalid = Tensor::scalar(cuda, -1_f32).unwrap();
    let zero = Tensor::scalar(cuda, 0_i64).unwrap();
    assert!(matches!(
        device_input.quantize_with(&invalid, &zero, None, None, DataType::I8),
        Err(Error::InvalidArgument { .. })
    ));

    let lhs = input.quantize(0.5, 1, DataType::I8).unwrap();
    let rhs = matrix(&[4, 2], vec![1., 2., 3., 4., 5., 6., 7., 8.])
        .quantize(0.25, 0, DataType::U8)
        .unwrap();
    let expected = lhs.quantized_matmul(&rhs, 0.5, 1, 0.25, 0).unwrap();
    let actual = lhs
        .to_device(cuda)
        .unwrap()
        .quantized_matmul(&rhs.to_device(cuda).unwrap(), 0.5, 1, 0.25, 0)
        .unwrap();
    assert_eq!(
        actual.to_vec::<f32>().unwrap(),
        expected.to_vec::<f32>().unwrap()
    );
}
