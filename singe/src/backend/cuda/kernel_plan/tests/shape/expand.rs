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
