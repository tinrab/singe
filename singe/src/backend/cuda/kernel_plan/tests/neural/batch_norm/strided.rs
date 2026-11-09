use super::*;

#[test]
fn test_cuda_eager_batch_norm_f32_materializes_strided_input_and_parameters() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = contiguous_shape([2_i64, 5, 3]);
    let parameter_shape = contiguous_shape([3_i64, 2]);
    let values = (0..30usize)
        .map(|value| (value as f32).mul_add(0.0625, -1.0))
        .collect::<Vec<_>>();
    let weight_values = vec![0.5_f32, 1.25, -0.75];
    let bias_values = vec![0.25_f32, -0.5, 0.75];
    let running_mean_values = vec![-0.25_f32, 0.5, 1.0];
    let running_var_values = vec![0.75_f32, 1.5, 2.25];
    let mut permuted_values = Vec::with_capacity(values.len());
    for batch in 0..2usize {
        for channel in 0..3usize {
            for spatial in 0..5usize {
                permuted_values.push(values[(batch * 5 + spatial) * 3 + channel]);
            }
        }
    }
    let expected = expected_batch_norm_f32(
        &permuted_values,
        Some(&weight_values),
        Some(&bias_values),
        Some(&running_mean_values),
        Some(&running_var_values),
        3,
        5,
        1.0e-5,
    );
    let Some(input) = maybe_cuda_f32_tensor(&shape, values) else {
        return;
    };
    let input = input.permute(&[0, 2, 1]).unwrap();
    assert!(!input.is_contiguous());
    let weight = strided_parameter(&parameter_shape, &weight_values);
    let bias = strided_parameter(&parameter_shape, &bias_values);
    let running_mean = strided_parameter(&parameter_shape, &running_mean_values);
    let running_var = strided_parameter(&parameter_shape, &running_var_values);
    assert!(!weight.is_contiguous());
    assert!(!bias.is_contiguous());
    assert!(!running_mean.is_contiguous());
    assert!(!running_var.is_contiguous());

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

    assert_eq!(output.shape().dimensions(), &[2_i64, 3, 5]);
    assert!(output.is_contiguous());
    assert_close!(&output.to_vec::<f32>().unwrap(), &expected, 1.0e-5);
}

fn strided_parameter(shape: &Shape, values: &[f32]) -> Tensor {
    let interleaved = values
        .iter()
        .flat_map(|value| [*value, 99.0_f32])
        .collect::<Vec<_>>();
    f32_tensor(shape, interleaved).narrow(1, 0, 1).unwrap()
}
