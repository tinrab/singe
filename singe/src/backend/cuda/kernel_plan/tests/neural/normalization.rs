use super::*;

#[test]
fn test_cuda_eager_layer_norm_f32_materializes_tail_feature_width() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let rows = 2usize;
    let cols = 129usize;
    let shape =
        Shape::create([i64::try_from(rows).unwrap(), i64::try_from(cols).unwrap()]).unwrap();
    let parameter_shape = Shape::create([i64::try_from(cols).unwrap()]).unwrap();
    let values = normalization_input_values(rows, cols);
    let weight_values = (0..cols)
        .map(|index| (index as f32).mul_add(0.0025, 0.75))
        .collect::<Vec<_>>();
    let bias_values = (0..cols)
        .map(|index| (index as f32).mul_add(0.001, -0.5))
        .collect::<Vec<_>>();
    let expected = expected_normalization_f32(
        Operation::LayerNorm,
        &values,
        rows,
        cols,
        Some(&weight_values),
        Some(&bias_values),
        1.0e-5,
    );
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };
    let weight = Tensor::from_vec(Device::Cuda(0), &parameter_shape, weight_values).unwrap();
    let bias = Tensor::from_vec(Device::Cuda(0), &parameter_shape, bias_values).unwrap();

    let output = input
        .layer_norm(
            &[i64::try_from(cols).unwrap()],
            Some(&weight),
            Some(&bias),
            1.0e-5,
        )
        .unwrap();

    assert_eq!(
        output.shape().dimensions(),
        &[i64::try_from(rows).unwrap(), i64::try_from(cols).unwrap()]
    );
    assert_close!(&output.to_vec::<f32>().unwrap(), &expected, 1.0e-4);
}

#[test]
fn test_cuda_eager_rms_norm_f32_uses_default_parameters() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let rows = 17usize;
    let cols = 8usize;
    let shape =
        Shape::create([i64::try_from(rows).unwrap(), i64::try_from(cols).unwrap()]).unwrap();
    let values = normalization_input_values(rows, cols);
    let expected =
        expected_normalization_f32(Operation::RmsNorm, &values, rows, cols, None, None, 1.0e-4);
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };

    let output = input
        .rms_norm(&[i64::try_from(cols).unwrap()], None, None, 1.0e-4)
        .unwrap();

    assert_eq!(
        output.shape().dimensions(),
        &[i64::try_from(rows).unwrap(), i64::try_from(cols).unwrap()]
    );
    assert_close!(&output.to_vec::<f32>().unwrap(), &expected, 1.0e-4);
}

#[test]
fn test_cuda_eager_normalization_f32_materializes_strided_input_and_parameter() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2_i64, 3, 5]).unwrap();
    let weight_shape = Shape::create([5_i64, 2]).unwrap();
    let values = (0..30usize)
        .map(|value| (value as f32).mul_add(0.125, -1.0))
        .collect::<Vec<_>>();
    let weight_values = (0..10usize)
        .map(|value| (value as f32).mul_add(0.05, 0.5))
        .collect::<Vec<_>>();
    let mut permuted_values = Vec::with_capacity(values.len());
    for channel in 0..3usize {
        for batch in 0..2usize {
            for dim in 0..5usize {
                permuted_values.push(values[(batch * 3 + channel) * 5 + dim]);
            }
        }
    }
    let mut permuted_weight_values = Vec::with_capacity(weight_values.len());
    for row in 0..2usize {
        for col in 0..5usize {
            permuted_weight_values.push(weight_values[col * 2 + row]);
        }
    }
    let expected = expected_normalization_f32(
        Operation::LayerNorm,
        &permuted_values,
        3,
        10,
        Some(&permuted_weight_values),
        None,
        1.0e-5,
    );
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };
    let weight = Tensor::from_vec(Device::Cuda(0), &weight_shape, weight_values).unwrap();
    let strided = input.permute(&[1, 0, 2]).unwrap();
    let weight = weight.permute(&[1, 0]).unwrap();
    assert!(!strided.is_contiguous());
    assert!(!weight.is_contiguous());

    let output = strided
        .layer_norm(&[2, 5], Some(&weight), None, 1.0e-5)
        .unwrap();

    assert_eq!(output.shape().dimensions(), &[3_i64, 2, 5]);
    assert_close!(&output.to_vec::<f32>().unwrap(), &expected, 1.0e-4);
}

#[test]
fn test_cuda_eager_normalization_f32_runs_wide_rows_on_the_general_kernel() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let wide_shape = Shape::create([1_i64, 1025]).unwrap();
    let wide_values = (0..1025usize)
        .map(|value| value as f32 * 0.001)
        .collect::<Vec<_>>();
    let wide = Tensor::from_vec(Device::Cuda(0), &wide_shape, wide_values).unwrap();
    let expected = wide
        .to_device(Device::Cpu)
        .unwrap()
        .rms_norm(&[1025], None, None, 1.0e-5)
        .unwrap();
    let actual = wide.rms_norm(&[1025], None, None, 1.0e-5).unwrap();
    assert_close!(
        &actual.to_vec::<f32>().unwrap(),
        &expected.to_vec::<f32>().unwrap()
    );
}
