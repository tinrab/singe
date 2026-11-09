use super::*;

#[test]
fn test_cuda_eager_repeat_kv_f32_materializes_tail_output() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2_i64, 3, 5, 7]).unwrap();
    let values = (0..210usize)
        .map(|value| (value as f32).mul_add(0.03125, -3.0))
        .collect::<Vec<_>>();
    let expected = expected_repeat_kv_f32(&values, 2, 3, 5, 7, 2);
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };

    let output = input.repeat_kv(2).unwrap();

    assert_eq!(output.shape().dimensions(), &[2_i64, 6, 5, 7]);
    assert_close!(&output.to_vec::<f32>().unwrap(), &expected, 1.0e-5);
}

#[test]
fn test_cuda_eager_repeat_kv_f32_materializes_strided_input() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2_i64, 5, 3, 7]).unwrap();
    let values = (0..210usize)
        .map(|value| (value as f32).mul_add(-0.015_625, 4.0))
        .collect::<Vec<_>>();
    let mut permuted_values = Vec::with_capacity(values.len());
    for batch in 0..2usize {
        for head in 0..3usize {
            for sequence in 0..5usize {
                for dim in 0..7usize {
                    permuted_values.push(values[((batch * 5 + sequence) * 3 + head) * 7 + dim]);
                }
            }
        }
    }
    let expected = expected_repeat_kv_f32(&permuted_values, 2, 3, 5, 7, 3);
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values)) else {
        return;
    };
    let input = input.permute(&[0, 2, 1, 3]).unwrap();
    assert!(!input.is_contiguous());

    let output = input.repeat_kv(3).unwrap();

    assert_eq!(output.shape().dimensions(), &[2_i64, 9, 5, 7]);
    assert_close!(&output.to_vec::<f32>().unwrap(), &expected, 1.0e-5);
}

#[test]
fn test_cuda_eager_repeat_kv_f32_handles_noop() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([1_i64, 2, 3, 4]).unwrap();
    let values = (0..24usize).map(|value| value as f32).collect::<Vec<_>>();
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values.clone()))
    else {
        return;
    };

    let output = input.repeat_kv(1).unwrap();

    assert_eq!(output.shape().dimensions(), &[1_i64, 2, 3, 4]);
    assert_close!(&output.to_vec::<f32>().unwrap(), &values);
}
