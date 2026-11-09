use super::*;

#[test]
fn test_cuda_eager_batch_norm_f32_inference_materializes_tail_output() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = contiguous_shape([2_i64, 3, 4, 7]);
    let parameter_shape = contiguous_shape([3_i64]);
    let values = (0..168usize)
        .map(|value| (value as f32).mul_add(0.03125, -2.75))
        .collect::<Vec<_>>();
    let weight_values = vec![0.5_f32, -1.25, 2.0];
    let bias_values = vec![0.25_f32, -0.75, 1.5];
    let running_mean_values = vec![-0.5_f32, 0.25, 1.0];
    let running_var_values = vec![0.75_f32, 2.5, 4.0];
    let expected = expected_batch_norm_f32(
        &values,
        Some(&weight_values),
        Some(&bias_values),
        Some(&running_mean_values),
        Some(&running_var_values),
        3,
        28,
        1.0e-5,
    );
    let Some(input) = maybe_cuda_f32_tensor(&shape, values) else {
        return;
    };
    let weight = f32_tensor(&parameter_shape, weight_values);
    let bias = f32_tensor(&parameter_shape, bias_values);
    let running_mean = f32_tensor(&parameter_shape, running_mean_values);
    let running_var = f32_tensor(&parameter_shape, running_var_values);

    let output = input
        .batch_norm(
            Some(&weight),
            Some(&bias),
            Some(&running_mean),
            Some(&running_var),
            false,
            0.1,
            1.0e-5,
        )
        .unwrap();

    assert_eq!(output.shape().dimensions(), &[2_i64, 3, 4, 7]);
    assert_close!(&output.to_vec::<f32>().unwrap(), &expected, 1.0e-5);
}

#[test]
fn test_cuda_eager_batch_norm_f32_inference_uses_default_parameters() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = contiguous_shape([3_i64, 2, 5]);
    let values = (0..30usize)
        .map(|value| (value as f32).mul_add(-0.125, 3.0))
        .collect::<Vec<_>>();
    let expected = expected_batch_norm_f32(&values, None, None, None, None, 2, 5, 0.01);
    let Some(input) = maybe_cuda_f32_tensor(&shape, values) else {
        return;
    };

    let output = input
        .batch_norm(None, None, None, None, false, 0.1, 0.01)
        .unwrap();

    assert_eq!(output.shape().dimensions(), &[3_i64, 2, 5]);
    assert_close!(&output.to_vec::<f32>().unwrap(), &expected, 1.0e-5);
}
