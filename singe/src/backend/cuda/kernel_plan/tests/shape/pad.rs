use super::*;

#[test]
fn test_cuda_eager_pad_f32_rank4_matches_expected() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([3_i64, 43]).unwrap();
    let values = (0..129usize)
        .map(|value| (value as f32).mul_add(0.25, -3.0))
        .collect::<Vec<_>>();
    let padding = [(1_i64, 2_i64), (2_i64, 1_i64)];
    let expected = expected_pad_f32(&values, &[3, 43], &padding, -9.0);
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };

    let output = input.pad(&padding, -9.0).unwrap();

    assert_eq!(output.shape().dimensions(), &[6_i64, 46]);
    assert_close!(&output.to_vec::<f32>().unwrap(), &expected, 1.0e-5);

    let rank3_shape = Shape::create([2_i64, 3, 11]).unwrap();
    let rank3_values = (0..66usize)
        .map(|value| (value as f32).mul_add(0.125, -4.0))
        .collect::<Vec<_>>();
    let rank3_padding = [(0_i64, 1_i64), (1_i64, 0_i64), (2_i64, 3_i64)];
    let mut rank3_permuted_values = Vec::with_capacity(rank3_values.len());
    for i in 0..11usize {
        for j in 0..2usize {
            for k in 0..3usize {
                rank3_permuted_values.push(rank3_values[j * 3 * 11 + k * 11 + i]);
            }
        }
    }
    let expected_rank3 = expected_pad_f32(&rank3_permuted_values, &[11, 2, 3], &rank3_padding, 2.5);
    let rank3_input = Tensor::from_vec(Device::Cuda(0), &rank3_shape, rank3_values).unwrap();
    let rank3_permuted = rank3_input.permute(&[2, 0, 1]).unwrap();
    assert!(!rank3_permuted.is_contiguous());

    let rank3_output = rank3_permuted.pad(&rank3_padding, 2.5).unwrap();

    assert_eq!(rank3_output.shape().dimensions(), &[12_i64, 3, 8]);
    assert_close!(
        &rank3_output.to_vec::<f32>().unwrap(),
        &expected_rank3,
        1.0e-5
    );
}

#[test]
fn test_cuda_eager_pad_f32_handles_noop_and_rank5() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2_i64, 3]).unwrap();
    let values = (0..6usize).map(|value| value as f32).collect::<Vec<_>>();
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values.clone()))
    else {
        return;
    };

    let noop = input.pad(&[(0, 0), (0, 0)], 7.0).unwrap();

    assert_eq!(noop.shape().dimensions(), &[2_i64, 3]);
    assert_close!(&noop.to_vec::<f32>().unwrap(), &values);

    let rank5_shape = Shape::create([2_i64, 2, 2, 2, 2]).unwrap();
    let rank5_values = (0..32usize).map(|value| value as f32).collect::<Vec<_>>();
    let rank5_input = Tensor::from_vec(Device::Cuda(0), &rank5_shape, rank5_values).unwrap();

    let rank5 = rank5_input
        .pad(&[(1, 0), (0, 0), (0, 0), (0, 0), (0, 0)], -1.0)
        .unwrap();
    assert_eq!(rank5.shape().dimensions(), &[3_i64, 2, 2, 2, 2]);
    let expected = [-1.0_f32; 16]
        .into_iter()
        .chain((0..32_u8).map(f32::from))
        .collect::<Vec<_>>();
    assert_close!(&rank5.to_vec::<f32>().unwrap(), &expected);
}
