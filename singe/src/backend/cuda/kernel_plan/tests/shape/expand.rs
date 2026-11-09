use super::*;

#[test]
fn test_cuda_eager_expand_f32_preserves_shared_tail_broadcast() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let input_shape = Shape::create([3_i64, 1]).unwrap();
    let output_shape = Shape::create([3_i64, 43]).unwrap();
    let values = vec![-2.0_f32, 0.5, 4.0];
    let expected = expected_expand_f32(&values, &[3, 1], &[3, 43]);
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &input_shape, values)) else {
        return;
    };

    let output = input.expand(&output_shape).unwrap();

    assert_eq!(output.shape().dimensions(), &[3_i64, 43]);
    assert_eq!(output.layout().strides(), &[1, 0]);
    assert_eq!(
        output.storage().storage_identity(),
        input.storage().storage_identity()
    );
    assert_close!(&output.to_vec::<f32>().unwrap(), &expected, 1.0e-5);
}

#[test]
fn test_cuda_eager_expand_f32_preserves_strided_input() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let input_shape = Shape::create([2_i64, 3, 1]).unwrap();
    let output_shape = Shape::create([3_i64, 2, 23]).unwrap();
    let values = (0..6usize)
        .map(|value| (value as f32).mul_add(0.25, -1.0))
        .collect::<Vec<_>>();
    let mut permuted_values = Vec::with_capacity(values.len());
    for i in 0..3usize {
        for j in 0..2usize {
            permuted_values.push(values[j * 3 + i]);
        }
    }
    let expected = expected_expand_f32(&permuted_values, &[3, 2, 1], &[3, 2, 23]);
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &input_shape, values)) else {
        return;
    };
    let input = input.permute(&[1, 0, 2]).unwrap();
    assert!(!input.is_contiguous());

    let output = input.expand(&output_shape).unwrap();

    assert_eq!(output.shape().dimensions(), &[3_i64, 2, 23]);
    assert_close!(&output.to_vec::<f32>().unwrap(), &expected, 1.0e-5);
}

#[test]
fn test_cuda_eager_expand_f32_preserves_noop_and_rank5_broadcast_views() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };

    let shape = Shape::create([2_i64, 3]).unwrap();
    let values = (0..6usize).map(|value| value as f32).collect::<Vec<_>>();
    let Some(input) = skip_if_stub(Tensor::from_vec(Device::Cuda(0), &shape, values.clone()))
    else {
        return;
    };

    let noop = input.expand(&shape).unwrap();

    assert_eq!(noop.shape().dimensions(), &[2_i64, 3]);
    assert_close!(&noop.to_vec::<f32>().unwrap(), &values);

    let rank5_input_shape = Shape::create([1_i64, 1, 1, 1, 1]).unwrap();
    let rank5_output_shape = Shape::create([1_i64, 1, 1, 1, 2]).unwrap();
    let rank5_input = Tensor::from_vec(Device::Cuda(0), &rank5_input_shape, vec![1.0_f32]).unwrap();

    let expanded = rank5_input.expand(&rank5_output_shape).unwrap();
    assert_eq!(expanded.layout().strides(), &[1, 1, 1, 1, 0]);
    assert_eq!(expanded.to_vec::<f32>().unwrap(), [1.0, 1.0]);
}
