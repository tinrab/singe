use super::*;

#[test]
fn test_cuda_eager_batch_norm_f32_training_updates_the_running_mean() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = contiguous_shape([2_i64, 3, 5]);
    let values = (0..30usize)
        .map(|value| (value as f32).mul_add(0.25, -1.0))
        .collect::<Vec<_>>();
    let Some(input) = maybe_cuda_f32_tensor(&shape, values) else {
        return;
    };
    let Some(mean) = maybe_cuda_f32_tensor(&contiguous_shape([3_i64]), vec![0.0; 3]) else {
        return;
    };

    input
        .batch_norm(None, None, Some(&mean), None, true, 0.1, 1.0e-5)
        .unwrap();
    // Channel `c` holds the values at flat indices `15n + 5c + s`, whose mean index is `5c + 9.5`.
    let expected = (0..3_u8)
        .map(|channel| 0.1 * f32::mul_add(f32::from(channel), 5.0, 9.5).mul_add(0.25, -1.0))
        .collect::<Vec<_>>();
    assert_close!(&mean.to_vec::<f32>().unwrap(), &expected, 1.0e-6);
}
